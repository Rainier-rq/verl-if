# intruction_system_prompt = '''You are a helpful assistant who rigorously follows all explicit and implicit constraints in user instructions. When the user provides an instruction containing multiple constraints, you first analyze how to best follow it. During the thinking process, you conduct a comprehensive constraint analysis and then develop execution strategies that fully comply with all requirements. Finally, you provide the user with an answer that adheres to all specified formats and rules. Both the thinking process and answer must be enclosed within the respective tags:
# <think>
# The thinking process for constraint analysis and execution strategies
# </think>

# <answer>
# An answer adhering to all specified formats and rules
# </answer>
# '''
EXAMPLE = """
Instruction: 
In this task, you're given an article and an answer. Your task is to generate the fill-in-the-blank question for this answer based on the given article. The blanks are always at the end of the question, and can be filled with more than one word or phrase. \n input : Article: In 1955, a man named Raymond Kroc entered a partnership with two brothers named Mcdonald. They operated a popular restaurant in California which sold food that was easy to prepare and serve quickly. Hamburgers, French fries, and cold drinks were the main foods on the limited menu. Kroc opened similar eating places under the same name, \"McDonald's\" and they were a quick success. He later took over the company and today it is one of the most famous and successful \"fast-food\" chains  in America and the world. Why was his idea so successful? Probably the most important reason was that his timing was right. In the 1950s, most married women stayed home to keep house and take care of their children. During the 1960s, the movement for equality  between men and women and an economy  that needed more families to have two money-earners resulted in many women returning to the workplace. This meant that they had less time and energy to do housework, so they depended more on fast-food restaurants. Single parents also have little time to spend in the kitchen. People living alone because of divorce  or a preference for a \"single lifestyle\" also depend on this type of food, since cooking for one is often more trouble than it is worth. Fast food is not part of the diet of all Americans. Another of the 1960s was called back-to-nature movement. More and more Americans based their diets on natural foods. This preference for natural foods continues to this day. These products can now be found not only in the special health food store but also in many supermarkets. \n Answer: the effect social and economic changes have on people's eating habits Your answer must contain exactly 4 bullet point in Markdown using the following format:\n* Bullet point one.\n* Bullet point two.\n.... answer with less than 70. Your answer must have at least 3 placeholders, wrapped in square brackets, such as [author]. Highlight at least 3 sections of your response in markdown such as *highlighted section*. Your answer should be in all capital letters, no lowercase letters allowed.

Output:
<think>
Alright, let's break down this task:\n\n1. I need to generate a fill-in-the-blank question based on a given article where the answer is already provided.\n2. The fill-in-the-blanks will be at the end of the question and can be more than one word/phrase.\n3. The article is about McDonald's rise and changes in American eating habits.\n4. The answer given is \"the effect social and economic changes have on people's eating habits.\"\n\nExplicit Constraints:\n- My response must contain exactly 4 bullet points in Markdown format.\n- My answer should be less than 70 words.\n- My answer must include at least 3 placeholders in square brackets.\n- I need to highlight at least 3 sections of my response in markdown.\n- My answer should be in ALL CAPITAL LETTERS.\n\nLet me start by thinking about what kind of fill-in-the-blank question would lead to the answer \"the effect social and economic changes have on people's eating habits.\"\n\nThe article discusses how McDonald's became successful, particularly due to timing with social changes like women entering the workforce, single parents having less time, and people living alone. It also mentions a back-to-nature movement affecting food choices. So the question should be about how these social and economic changes influenced eating habits.\n\nPotential fill-in-the-blank question: \"The article primarily discusses how the rise of fast food restaurants like McDonald's and health food stores was influenced by [social and economic changes], which demonstrates [the impact] of these changes on [people's eating habits].\"\n\nNow I need to format this in ALL CAPS, with bullet points, and highlight certain sections.\n\nLet me draft the answer:\n\n* THE ARTICLE PRIMARILY DISCUSSES HOW THE RISE OF FAST FOOD RESTAURANTS LIKE *MCDONALD'S* WAS INFLUENCED BY [SOCIAL TRENDS].\n* THESE TRENDS INCLUDED *WOMEN ENTERING THE WORKFORCE* AND CHANGES IN FAMILY STRUCTURE SUCH AS [SINGLE PARENTING].\n* THE SUCCESS OF FAST FOOD CHAINS AND ALSO *HEALTH FOOD STORES* DEMONSTRATES [EVOLVING CONSUMER NEEDS].\n* THE TEXT ILLUSTRATES HOW THESE FACTORS COMBINED TO CHANGE [AMERICAN EATING HABITS].\n\nLet me check if this meets all the constraints:\n- Contains exactly 4 bullet points in Markdown format ✓\n- Is in ALL CAPITAL LETTERS ✓\n- Has at least 3 placeholders in square brackets: [SOCIAL TRENDS], [SINGLE PARENTING], [EVOLVING CONSUMER NEEDS], [AMERICAN EATING HABITS] ✓\n- Has at least 3 highlighted sections: *MCDONALD'S*, *WOMEN ENTERING THE WORKFORCE*, *HEALTH FOOD STORES* ✓\n- Word count: approximately 63 words, which is less than 70 ✓\n\nHowever, I need to make sure the fill-in-the-blank question actually leads to the answer \"the effect social and economic changes have on people's eating habits.\" Let me refine this to make it clearer:\n\n* THE ARTICLE PRIMARILY DISCUSSES HOW *FAST FOOD RESTAURANTS* LIKE MCDONALD'S BECAME SUCCESSFUL DUE TO [CHANGING SOCIAL PATTERNS].\n* FACTORS SUCH AS *WOMEN ENTERING THE WORKFORCE* AND THE RISE OF [SINGLE-PARENT HOUSEHOLDS] CONTRIBUTED.\n* THE TEXT ALSO MENTIONS *HEALTH FOOD TRENDS* AS ANOTHER RESPONSE TO [SOCIETAL SHIFTS].\n* OVERALL, THE PASSAGE DEMONSTRATES [THE EFFECT SOCIAL AND ECONOMIC CHANGES HAVE ON PEOPLE'S EATING HABITS].\n\nThis better aligns with the expected answer, as the last bullet point's placeholder directly contains the answer. But I realized the last placeholder essentially gives away the entire answer, which isn't how fill-in-the-blank questions typically work. Let me adjust:\n\n* THE ARTICLE PRIMARILY DISCUSSES HOW *FAST FOOD RESTAURANTS* LIKE MCDONALD'S BECAME SUCCESSFUL DUE TO [TIMING].\n* FACTORS SUCH AS *WOMEN ENTERING THE WORKFORCE* AND THE RISE OF [SINGLE-PARENT HOUSEHOLDS] CONTRIBUTED.\n* THE TEXT ALSO MENTIONS *HEALTH FOOD TRENDS* AS ANOTHER RESPONSE TO [SOCIETAL SHIFTS].\n* OVERALL, THE PASSAGE DEMONSTRATES WHAT?\n\nThis makes it a proper fill-in-the-blank question where the answer would be \"the effect social and economic changes have on people's eating habits.\"\n\nHowever, I see that the instructions want the blanks to be at the end of the question. So I need to restructure this again.\n\nLet's try another approach:\n\n* THE ARTICLE DESCRIBES HOW *FAST FOOD RESTAURANTS* LIKE MCDONALD'S SUCCEEDED DURING A TIME OF [SOCIAL CHANGE].\n* *WOMEN ENTERING THE WORKFORCE* AND CHANGES IN FAMILY STRUCTURE CREATED DEMAND FOR [CONVENIENT DINING OPTIONS].\n* THE RISE OF THE *BACK-TO-NATURE MOVEMENT* SIMULTANEOUSLY CREATED MARKETS FOR [HEALTH FOOD].\n* THESE EXAMPLES ILLUSTRATE [THE EFFECT SOCIAL AND ECONOMIC CHANGES HAVE ON PEOPLE'S EATING HABITS].\n\nThis structure better places the blanks at the end of statements, with the final blank containing the answer. Let me check the constraints again:\n\n- Contains exactly 4 bullet points in Markdown format ✓\n- Is in ALL CAPITAL LETTERS ✓\n- Has at least 3 placeholders in square brackets ✓\n- Has at least 3 highlighted sections ✓\n- Word count: approximately 66 words, which is less than 70 ✓\n\nBut I now realize I've been misunderstanding the task. I need to create a single fill-in-the-blank question where the answer is \"the effect social and economic changes have on people's eating habits.\" The bullet points should be part of my answer format, not the question itself.\n\nLet me reframe this:\n\nThe question should be something like: \"According to the article, the success of fast-food chains like McDonald's and the growth of health food stores demonstrates [the effect social and economic changes have on people's eating habits].\"\n\nNow, I need to present this as my answer in the required format. Let me create the final answer:
</think>

<answer>
* THE ARTICLE EXPLAINS HOW *FAST FOOD RESTAURANTS* BECAME SUCCESSFUL DUE TO [SOCIETAL CHANGES].\n* THE TEXT DESCRIBES HOW *WOMEN ENTERING THE WORKFORCE* CREATED DEMAND FOR [CONVENIENT DINING].\n* THE PASSAGE ALSO MENTIONS HOW THE *BACK-TO-NATURE MOVEMENT* INFLUENCED [FOOD PREFERENCES].\n* ACCORDING TO THE ARTICLE, THE RISE OF MCDONALD'S AND HEALTH FOOD STORES DEMONSTRATES [THE EFFECT SOCIAL AND ECONOMIC CHANGES HAVE ON PEOPLE'S EATING HABITS].
</answer>
""".strip()

instruction_system_prompt = """You are a meticulous assistant who precisely adheres to all explicit and implicit constraints in user instructions. When presented with complex, multi-constraint tasks, you follow a structured approach:

FORMAT: Your response must be structured with two distinct sections:
<think> Your detailed analytical process and strategy development </think>
<answer> Your final solution that adheres to all requirements </answer>

In your <think> section, employ these strategic approaches:
1. Constraint Analysis: Thoroughly examine each explicit and implicit constraint, identifying their interdependencies and potential conflicts.
2. Validation Strategy: Draft preliminary answers, then methodically verify compliance with each constraint, making iterative refinements as needed.
3. Sequencing Logic: Determine the optimal order for addressing constraints, prioritizing foundational requirements before tackling dependent ones.
4. Edge Case Consideration: Proactively identify boundary conditions and exceptions that might challenge constraint compliance.
5. Coherence Check: Ensure your solution maintains internal consistency while satisfying all requirements simultaneously.
6. Format Verification: Confirm your response adheres to all specified formatting and structural guidelines.

In your <answer> section, deliver a solution that precisely implements all requirements while maintaining natural flow and coherence. Your final response must satisfy all constraints without drawing attention to the mechanics of constraint management itself.

Here is an example: 

{example}
""".format(example = EXAMPLE).strip()

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        # answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = "",
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.system_prompt=instruction_system_prompt
        # self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        # messages = [{"role": "user", "content": row_dict[self.prompt_key] +  '\n\n' + 'Remember: We will check for the "<think>.*</think><answer>.*</answer>" pattern to ensure compliance.'}]
        messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
        # if self.system_prompt:
        #     messages.insert(0, {"role": "system", "content": self.system_prompt})

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


        if self.image_key in row_dict:
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["multi_modal_data"] = {
                "image": [
                    process_image(image, self.max_pixels, self.min_pixels) for image in row_dict.pop(self.image_key)
                ]
            }
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        instruction_id_list = row_dict.pop('instruction_id_list')
        kwargs = row_dict.pop('kwargs')
        constraints= row_dict.pop('constraint')
        row_dict["ground_truth"] = {
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
            "prompt": row_dict[self.prompt_key],
            "constraints": constraints
        }
        return row_dict
