import json
import random
from collections import defaultdict

def process_jsonl_file(input_path, output_path, target_sample_size):
    # 读取数据并分类
    light_items = []
    light_choice_items = []
    other_items = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            instruction_ids = item.get("instruction_id_list", [])
            
            if "light" in instruction_ids and "light_choice" not in instruction_ids:
                light_items.append(item)
            elif "light_choice" in instruction_ids:
                light_choice_items.append(item)
            else:
                other_items.append(item)
    
    # 计算需要采样的总数
    total_available = len(light_items) + len(light_choice_items)
    sample_size = min(target_sample_size, total_available)
    print(f"Total available: {total_available}, Sampling: {sample_size}")
    
    # 计算比例
    light_ratio = len(light_items) / total_available
    light_choice_ratio = len(light_choice_items) / total_available
    
    # 计算每类采样数量
    light_sample_size = int(round(sample_size * light_ratio))
    light_choice_sample_size = sample_size - light_sample_size
    
    # 确保不超过实际数量
    light_sample_size = min(light_sample_size, len(light_items))
    light_choice_sample_size = min(light_choice_sample_size, len(light_choice_items))
    
    # 采样
    sampled_light = random.sample(light_items, light_sample_size) if light_items else []
    sampled_light_choice = random.sample(light_choice_items, light_choice_sample_size) if light_choice_items else []
    
    # 合并所有数据
    final_data = other_items + sampled_light + sampled_light_choice 
    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processing completed. Total items written: {len(final_data)}")
    print(f"Sampled light: {len(sampled_light)}, Sampled light_choice: {len(sampled_light_choice)}, Other items: {len(other_items)}")

# 使用示例
input_file = "/fs-computility/wangxuhong/renqingyu/qingyu/verl/data/train_all_scaler_processed.jsonl"  # 替换为你的输入文件路径
output_file = "/fs-computility/wangxuhong/renqingyu/qingyu/verl/data/train_all_scaler_processed_2w.jsonl"  # 替换为你的输出文件路径
target_sample_size = 20000 - 13570  # 6430

process_jsonl_file(input_file, output_file, target_sample_size)