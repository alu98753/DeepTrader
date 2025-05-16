import numpy as np
import pandas as pd

data = np.load('./data/DJIA/DJIA_CHLVMP.npy') # 确保这是您包含NaN的原始文件

num_stocks, num_days, num_features = data.shape
filled_data = np.copy(data)

for i in range(num_stocks):
    for j in range(num_features):
        # 将每个特征的时间序列转换为 pandas Series 以便使用 ffill/bfill
        stock_feature_series = pd.Series(data[i, :, j])
        filled_series = stock_feature_series.ffill().bfill()
        filled_data[i, :, j] = filled_series.to_numpy()

# 检查是否仍有NaN（如果某股票的某特征完全是NaN，则ffill/bfill无效）
remaining_nans = np.isnan(filled_data).sum()
print(f"NaNs remaining after ffill and bfill: {remaining_nans}")

if remaining_nans > 0:
    print("Warning: NaNs still present. These might be entire series of NaNs for certain stock/feature pairs.")
    # 对于完全是NaN的特征，需要更具体的策略
    # 例如，对于Volume，如果全为NaN，可以考虑填充0
    # for i in range(num_stocks):
    #     # 假设第3个特征是Volume (索引从0开始)
    #     if np.all(np.isnan(filled_data[i, :, 3])): # 假设第4个特征是Volume
    #         print(f"Stock {i}, feature 3 (Volume?) is all NaNs. Filling with 0.")
    #         filled_data[i, :, 3] = 0.0
    # 对于价格相关的，如果全NaN，问题较大，可能需要用0填充（如果后续会标准化且0是中性值）或从数据源修复
    # 这是一个临时的、可能不太理想的最终手段，以确保没有NaN传递给模型：
    filled_data = np.nan_to_num(filled_data, nan=0.0) # 将剩余NaN替换为0 (请谨慎评估影响)
    print(f"NaNs after final np.nan_to_num(0.0) fill: {np.isnan(filled_data).sum()}")


# 转换为 float32 并保存
filled_data_float32 = filled_data.astype(np.float32)
np.save('./data/DJIA/stocks_data.npy', filled_data_float32)
print("Filled data saved as stocks_data.npy (float32)")