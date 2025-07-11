# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import  r1v_compute_score,gsm8k_compute_score,instruction_compute_score,instruction_val_compute_score


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.mode = ""
        if compute_score == "instruction":
            self.compute_score = instruction_compute_score
            self.mode = "train"
        elif compute_score == "instruction_val":
            self.compute_score = instruction_val_compute_score
            self.mode = "val"
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "gsm8k":
            self.compute_score = gsm8k_compute_score
        
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # reward_tensor.shape=torch.Size([60, 4096])
        ac_count=0
        c_count=0
        score_l=0
        already_print=0
        # len(data) = 60
        
        constraint_response_lst = []

        for i in range(len(data)):
            
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["ground_truth"]

            constraints = ground_truth['instruction_id_list']

 
            
            constraint_response_lst.append({"cc": constraints,'rl': valid_response_length})
            

            


            if self.mode == "train":
                score = self.compute_score(response_str, ground_truth)
            elif self.mode == "val":
                all_right,a_len_c,len_c,score = self.compute_score(response_str, ground_truth)
                ac_count+=a_len_c
                c_count+=len_c
                score_l +=all_right

            reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                # print("[ground_truth]", ground_truth)
                print("[reward score]", score)

       

        if self.mode=='train':
            return reward_tensor, constraint_response_lst
        elif self.mode=='val':
            score_c =ac_count/c_count
            score_l = score_l/len(data)
            return reward_tensor,score_c,score_l
