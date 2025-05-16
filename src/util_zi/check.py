
import numpy as np
data_prefix = './data/DJIA/'

stocks_data = np.load(data_prefix + 'market_data.npy')

# 在 run.py 載入 stocks_data 之後
print(f"DEBUG: stocks_data.npy loaded. Shape: {stocks_data.shape}, Dtype: {stocks_data.dtype}")
if np.isnan(stocks_data).any():
    print("WARNING: stocks_data.npy CONTAINS NaN VALUES!")
    # 可以更詳細地打印 NaN 的數量或位置
    print(f"Total NaN count in stocks_data: {np.isnan(stocks_data).sum()}")
else:
    print("DEBUG: stocks_data.npy does not contain NaN values.")