import numpy as np
import pandas as pd

# 定义 ROR.npy 文件的路径
ror_file_path = './data/DJIA/ROR.npy'
# 定义处理后文件的保存路径 (建议先保存为新文件名以作区分)
ror_filled_file_path = './data/DJIA/ROR_filled.npy'

# 1. 载入 ROR.npy 数据
try:
    ror_data = np.load(ror_file_path)
    print(f"Original ROR data loaded. Shape: {ror_data.shape}, Dtype: {ror_data.dtype}")
    original_nan_count = np.isnan(ror_data).sum()
    print(f"Original NaN count in ROR data: {original_nan_count}")

    if original_nan_count == 0:
        print("ROR data contains no NaN values. No filling needed.")
        # 如果不需要填充，但仍希望统一数据类型为 float32，可以这样做：
        # ror_data_float32 = ror_data.astype(np.float32)
        # np.save(ror_filled_file_path, ror_data_float32)
        # print(f"ROR data (already clean) saved as {ror_filled_file_path} with dtype float32.")
    else:
        # 2. 填充 NaN 值
        num_stocks, num_days = ror_data.shape
        filled_ror_data = np.copy(ror_data) # 操作副本

        for stock_idx in range(num_stocks):
            # 将每支股票的收益率时间序列转换为 pandas Series
            ror_series = pd.Series(ror_data[stock_idx, :])
            
            # 先前向填充，再后向填充
            filled_series = ror_series.ffill().bfill()
            
            filled_ror_data[stock_idx, :] = filled_series.to_numpy()

        remaining_nans = np.isnan(filled_ror_data).sum()
        print(f"NaNs remaining in ROR data after ffill and bfill: {remaining_nans}")

        if remaining_nans > 0:
            print("WARNING: ROR data still contains NaNs after ffill and bfill.")
            print("This might happen if an entire stock's ROR series is NaN.")
            print("Applying np.nan_to_num(nan=0.0) as a final step to remove all NaNs.")
            # 对于收益率，如果某股票的整个历史收益率都是NaN，用0填充可能是一个中性的选择，
            # 代表没有收益也没有损失，但这仍需谨慎评估其对模型的影响。
            filled_ror_data = np.nan_to_num(filled_ror_data, nan=0.0)
            final_nan_check = np.isnan(filled_ror_data).sum()
            print(f"NaNs after final np.nan_to_num(0.0) fill: {final_nan_check}")
        
        # 3. 转换为 float32 并保存
        filled_ror_data_float32 = filled_ror_data.astype(np.float32)
        np.save(ror_filled_file_path, filled_ror_data_float32)
        print(f"Filled ROR data saved to {ror_filled_file_path}. Shape: {filled_ror_data_float32.shape}, Dtype: {filled_ror_data_float32.dtype}")

except FileNotFoundError:
    print(f"Error: {ror_file_path} not found.")
except Exception as e:
    print(f"An error occurred: {e}")