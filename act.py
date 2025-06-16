import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('/fs-computility/ai-shen/renqingyu/verl/data/train.parquet')

# 输出行数
print(f"Number of elements (rows): {len(df)}")
