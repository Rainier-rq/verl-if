import pandas as pd

# 读取 JSONL 文件，注意每行是一个独立的 JSON 对象
df = pd.read_json('/fs-computility/wangxuhong/renqingyu/qingyu/verl/data/train_all_scaler_processed_2w_dp.jsonl', orient='records', lines=True)

# 将 DataFrame 转换为 Parquet 文件
df.to_parquet('/fs-computility/wangxuhong/renqingyu/qingyu/verl/data/train_all_scaler_processed_2w_dp.parquet', engine='pyarrow')  # 或使用 'fastparquet' 作为引擎
