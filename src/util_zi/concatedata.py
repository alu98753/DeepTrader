import numpy as np
import os

# 設定檔案路徑
data_prefix = './data/DJIA/'
adl_file = os.path.join(data_prefix, 'ADL.npy')
adr_file = os.path.join(data_prefix, 'ADR.npy')
obos_file = os.path.join(data_prefix, 'OBOS.npy')
dji_index_file = os.path.join(data_prefix, 'DJI_index.npy')
output_market_data_file = os.path.join(data_prefix, 'market_data.npy')

# 載入 .npy 檔案
adl_data = np.load(adl_file)
adr_data = np.load(adr_file)
obos_data = np.load(obos_file)
dji_index_data = np.load(dji_index_file)

# 假設每個檔案的形狀都是 [num_days, 1]
# 我們需要將它們合併成 [num_days, 4]
# 作者提供的順序可能是重要的，我們按照 ADL, ADR, OBOS, DJI_index 的順序合併
# 如果您的 `hyper.json` 中 `in_features[1]` (即 num_MSU_features) 是 4，
# 並且每個檔案確實是 [num_days, 1]，這樣合併後維度就會是 [num_days, 4]

# 檢查第一個維度 (num_days) 是否一致
if not (adl_data.shape[0] == adr_data.shape[0] == obos_data.shape[0] == dji_index_data.shape[0]):
    raise ValueError("所有市場數據檔案的天數 (第一個維度) 必須相同！")

# 檢查第二個維度是否為 1
if not (adl_data.shape[1] == 1 and adr_data.shape[1] == 1 and obos_data.shape[1] == 1 and dji_index_data.shape[1] == 1):
    # 如果不是 [num_days, 1]，但仍希望每個檔案作為一個特徵，可能需要 reshape
    # 例如，如果它們是 [num_days]，則需要 reshape 成 [num_days, 1]
    # adl_data = adl_data.reshape(-1, 1) # 等等
    # 但您提到 dim1 == 1，所以這一步應該不需要
    print("警告：一個或多個市場數據檔案的第二個維度不為 1，請確認形狀。")


# 沿著第二個軸 (axis=1，即特徵軸) 合併
market_data_combined = np.concatenate((adl_data, adr_data, obos_data, dji_index_data), axis=1)

# 檢查合併後的形狀
print(f"ADL data shape: {adl_data.shape}")
print(f"ADR data shape: {adr_data.shape}")
print(f"OBOS data shape: {obos_data.shape}")
print(f"DJI Index data shape: {dji_index_data.shape}")
print(f"合併後的 market_data shape: {market_data_combined.shape}")

# 驗證 num_MSU_features 是否為 4 (根據 hyper.json in_features[1] 通常是4)
if market_data_combined.shape[1] == 4:
    print("合併後的特徵數量為 4，符合預期。")
else:
    print(f"警告：合併後的特徵數量為 {market_data_combined.shape[1]}，請檢查 hyper.json 中的 in_features[1] 設定。")

# 保存合併後的檔案
np.save(output_market_data_file, market_data_combined)
print(f"已將合併後的市場數據保存至: {output_market_data_file}")