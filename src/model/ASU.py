import math

import torch
import torch.nn as nn


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(GraphConvNet, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = nn.functional.dropout(h, self.dropout, training=self.training)
        return h


class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_nodes, in_features, in_len):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Linear(in_len, 1, bias=False)
        self.W2 = nn.Linear(in_features, in_len, bias=False)
        self.W3 = nn.Linear(in_features, 1, bias=False)
        self.V = nn.Linear(num_nodes, num_nodes)

        self.bn_w1 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w3 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w2 = nn.BatchNorm1d(num_features=num_nodes)

    def forward(self, inputs):
        # inputs: (batch, num_features, num_nodes, window_len)
        part1 = inputs.permute(0, 2, 1, 3)
        part2 = inputs.permute(0, 2, 3, 1)
        part1 = self.bn_w1(self.W1(part1).squeeze(-1))
        part1 = self.bn_w2(self.W2(part1))
        part2 = self.bn_w3(self.W3(part2).squeeze(-1)).permute(0, 2, 1)  #
        S = torch.softmax(self.V(torch.relu(torch.bmm(part1, part2))), dim=-1)
        return S


class SAGCN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, window_len,
                 dropout=0.3, kernel_size=2, layers=4, supports=None,
                 spatial_bool=True, addaptiveadj=True, aptinit=None):

        super(SAGCN, self).__init__()
        self.dropout = dropout
        self.layers = layers
        if spatial_bool:
            self.gcn_bool = True
            self.spatialattn_bool = True
        else:
            self.gcn_bool = False
            self.spatialattn_bool = False
        self.addaptiveadj = addaptiveadj

        self.tcns = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.sans = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.supports = supports

        self.start_conv = nn.Conv2d(in_features, hidden_dim, kernel_size=(1, 1))
        self.bn_start = nn.BatchNorm2d(hidden_dim)

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool and addaptiveadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec = nn.Parameter(torch.randn(num_nodes, 1), requires_grad=True)
                self.supports_len += 1

            else:
                raise NotImplementedError

        additional_scope = kernel_size - 1
        a_s_records = []
        dilation = 1
        for l in range(layers):
            time_kernel_width = kernel_size # kernel_size 是 SAGCN __init__ 的參數, 例如 hyper.json 中的 2

            # 時間卷積層 (TCN part)
            # 輸入 x 的形狀是 (batch, hidden_dim, num_nodes, current_seq_len)
            # 我們希望對每個節點的特徵序列獨立地在時間維度上進行卷積
            # kernel_size=(1, time_kernel_width) 表示卷積核高度為1（不跨節點的特徵通道），寬度為 time_kernel_width（時間步）
            # dilation=(1, dilation) 表示只在時間維度上進行膨脹
            # padding: 為了使殘差連接 x = x + residual[:, :, :, -x.shape[3]:] 成立，
            # TCN層的輸出時間維度長度需要與 residual 切片後的長度匹配。
            # 如果 time_kernel_width = K 和 dilation = D，標準無填充 Conv2d 的輸出長度會減少 D*(K-1)。
            # 這裡我們不加 padding，讓長度自然減少，殘差連接時會進行匹配。
            self.tcns.append(nn.Sequential(
                nn.Conv2d(in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=(1, time_kernel_width), # (kernel_height, kernel_width)
                        dilation=(1, dilation)),           # (dilation_height, dilation_width)
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm2d(hidden_dim) # BatchNorm2d 適用於4D輸入
            ))

            self.residual_convs.append(nn.Conv2d(in_channels=hidden_dim,
                                                out_channels=hidden_dim,
                                                kernel_size=(1, 1))) # 1x1 Conv2d 用於殘差連接
            self.bns.append(nn.BatchNorm2d(hidden_dim))

            if self.gcn_bool:
                self.gcns.append(GraphConvNet(hidden_dim, hidden_dim, dropout, support_len=self.supports_len))

            dilation *= 2
            a_s_records.append(additional_scope)
            receptive_field += additional_scope
            additional_scope *= 2

        self.receptive_field = receptive_field
        if self.spatialattn_bool:
            for i in range(layers):
                self.sans.append(SpatialAttentionLayer(num_nodes, hidden_dim, receptive_field - a_s_records[i]))
                receptive_field -= a_s_records[i]

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)  # [batch, feature, stocks, length]
        in_len = X.shape[3]
        if in_len < self.receptive_field:
            x = nn.functional.pad(X, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = X
        assert not torch.isnan(x).any()

        x = self.bn_start(self.start_conv(x))
        new_supports = None
        if self.gcn_bool and self.addaptiveadj and self.supports is not None:
            adp_matrix = torch.softmax(torch.relu(torch.mm(self.nodevec, self.nodevec.t())), dim=0)
            new_supports = self.supports + [adp_matrix]

        for i in range(self.layers):
            residual = self.residual_convs[i](x)
            x = self.tcns[i](x)
            if self.gcn_bool and self.supports is not None:
                if self.addaptiveadj:
                    x = self.gcns[i](x, new_supports)
                else:
                    x = self.gcns[i](x, self.supports)

            if self.spatialattn_bool:
                attn_weights = self.sans[i](x)
                x = torch.einsum('bnm, bfml->bfnl', (attn_weights, x))

            x = x + residual[:, :, :, -x.shape[3]:]

            x = self.bns[i](x)

        # (batch, num_nodes, hidden_dim)
        return x.squeeze(-1).permute(0, 2, 1)


class LiteTCN(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers, kernel_size=2, dropout=0.4):
        super(LiteTCN, self).__init__()
        self.num_layers = num_layers
        self.tcns = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.start_conv = nn.Conv1d(in_features, hidden_size, kernel_size=1)
        self.end_conv = nn.Conv1d(hidden_size, 1, kernel_size=1)

        receptive_field = 1
        additional_scope = kernel_size - 1
        dilation = 1
        for l in range(num_layers):
            tcn_sequence = nn.Sequential(nn.Conv1d(in_channels=hidden_size,
                                                   out_channels=hidden_size,
                                                   kernel_size=kernel_size,
                                                   dilation=dilation),
                                         nn.BatchNorm1d(hidden_size),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         )

            self.tcns.append(tcn_sequence)

            self.bns.append(nn.BatchNorm1d(hidden_size))

            dilation *= 2
            receptive_field += additional_scope
            additional_scope *= 2
        self.receptive_field = receptive_field

    def forward(self, X):
        X = X.permute(0, 2, 1)
        in_len = X.shape[2]
        if in_len < self.receptive_field:
            x = nn.functional.pad(X, (self.receptive_field - in_len, 0))
        else:
            x = X

        x = self.start_conv(x)

        for i in range(self.num_layers):
            residual = x
            assert not torch.isnan(x).any()
            x = self.tcns[i](x)
            assert not torch.isnan(x).any()
            x = x + residual[:, :, -x.shape[-1]:]

            x = self.bns[i](x)
        assert not torch.isnan(x).any()
        x = self.end_conv(x)

        return torch.sigmoid(x.squeeze())


class ASU(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, window_len,
                 dropout=0.3, kernel_size=2, layers=4, supports=None,
                 spatial_bool=True, addaptiveadj=True, aptinit=None):
        super(ASU, self).__init__()
        self.sagcn = SAGCN(num_nodes, in_features, hidden_dim, window_len, dropout, kernel_size, layers,
                           supports, spatial_bool, addaptiveadj, aptinit)
        self.linear1 = nn.Linear(hidden_dim, 1)

        # 原來的 bn1 和未使用的層:
        # self.bn1 = nn.BatchNorm1d(num_features=num_nodes)
        # self.in1 = nn.InstanceNorm1d(num_features=num_nodes) # 未使用
        # self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, ) # 未使用

        # 修改後 (使用 LayerNorm):
        self.ln1 = nn.LayerNorm(hidden_dim) # LayerNorm作用在最後一個維度 (hidden_dim)

        self.hidden_dim = hidden_dim # 這個似乎也沒在 ASU 中直接使用，但在 SAGCN 中可能間接相關

# --- 在 ASU 類的 forward 方法中 ---
    def forward(self, inputs, mask):
        """
        inputs: [batch, num_stock, window_len, num_features]
        mask: [batch, num_stock]
        outputs: [batch, scores]
        """
        x = self.ln1(self.sagcn(inputs)) # 或者 self.bn1，取決於您最終的選擇
        x = self.linear1(x).squeeze(-1)
        
        unmasked_score = torch.sigmoid(x) # Sigmoid 的原始輸出

        if mask is not None:
            # 使用 torch.where 避免原地修改
            # masked_score = torch.where(mask, torch.tensor(-math.inf, device=x.device, dtype=x.dtype), unmasked_score)
            # torch.where 在 PyTorch 2.x 中可能對 -math.inf 的梯度處理有更嚴格的要求，
            # 一個更安全且對於后续 softmax 更常見的做法是使用一個非常小的數值代替 -inf，
            # 或者確保 mask 的部分在 softmax 時得到極小概率。
            # 鑒於 DeepTrader 的邏輯是 topk + softmax，-math.inf 是為了確保不被選中。
            # 讓我們堅持使用 -math.inf，但用 torch.where。
            # 需要確保 -math.inf 被正確地轉換為與 unmasked_score 相同的 device 和 dtype。
            # PyTorch 1.3.1 可能對此更寬容，2.5.1 可能需要更明確。
            
            # 創建一個和 unmasked_score 同 device 同 dtype 的 -inf 張量
            neg_inf_tensor = torch.tensor(-math.inf, device=unmasked_score.device, dtype=unmasked_score.dtype)
            score_after_mask = torch.where(mask, neg_inf_tensor, unmasked_score)
        else:
            score_after_mask = unmasked_score
            
        return score_after_mask
