from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
import json
import re
from models.model import AutoModelForCausalLMWithScalarHead, AutoModelForCausalLMWithScalarHeadODIN


# Set CUDA device for Flask app logger
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = Flask(__name__)

# -------------------- Load reward model --------------------

def load_reward_model():
    from transformers import AutoTokenizer
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "/fs-computility/ai-shen/shared/huggingface/models/meta-llama/llama3-8b-inst"
    
    reward_model = AutoModelForCausalLMWithScalarHead.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load checkpoint
    checkpoint_path = "/fs-computility/ai-shen/shared/rqy/models/reward_gemma2-2b_ultrafb_bin/latest_hf.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    reward_model.load_state_dict(state_dict)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return reward_model.eval(), tokenizer, device

reward_model, reward_tokenizer, reward_device = load_reward_model()



# -------------------- Reward Evaluation --------------------

def reward_correct(answer,inst):
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, answer, re.DOTALL)
    if not answer_match:
        return 0
    
    # 组织输入
    input_text = json.dumps([
        {"content": inst, "role": "user"},
        {"content": answer_match.group(1), "role": "assistant"}
    ], ensure_ascii=False)



    # Tokenize
    inputs = reward_tokenizer(
        input_text,
        max_length=2048,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(reward_device)
    attention_mask = inputs['attention_mask'].to(reward_device)

    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')

    # Forward
    with torch.no_grad():
        rewards = reward_model(input_ids, attention_mask=attention_mask)
        masks = (input_ids != reward_tokenizer.pad_token_id).to(reward_device)
        last_token_idx = masks.sum(dim=1) - 1
        last_token_reward = rewards.gather(dim=1, index=last_token_idx.unsqueeze(1)).squeeze(-1)
        
    record = {
        "question": inst,
        "answer": answer_match.group(1),
        "reward": last_token_reward.item()
    }
    # x = torch.tensor(last_token_reward.item())
    # normalized_x = torch.sigmoid(x)

    # return float(normalized_x)
    # with open("reward_records.jsonl", "a", encoding="utf-8") as f:
    #     f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return last_token_reward.item()

# -------------------- Flask Endpoint --------------------

@app.route('/generate', methods=['POST'])
def generate():
    try:
        print("Raw request data:", request.data)  # 查看原始数据
        print("Headers:", request.headers)  
        #data = request.json
        data = json.loads(request.data)
        reward = reward_correct(data['answer'],data['prompt'])
        # x = torch.tensor(reward)  # 输入是 Tensor 类型
        # output_tensor = torch.sigmoid(x)  # 应用 sigmoid
        # reward = output_tensor.item()
        print("**********************")
        print(reward)
        print("**********************")
        return jsonify({'reward': reward})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# -------------------- Run App --------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=False)
