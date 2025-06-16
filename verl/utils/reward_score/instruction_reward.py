
import re
from modules import instructions_registry
from mathruler.grader import extract_boxed_content
import requests
import json
import inspect

def compute_score(expr: str, gt: list) -> float:
    positions = []
    idx = 0
    while idx < len(expr):
        if expr.startswith("boxed{", idx):
            start = idx + len("boxed{")
            stack = ["{"]
            i = start
            while i < len(expr) and stack:
                if expr[i] == "{":
                    stack.append("{")
                elif expr[i] == "}":
                    stack.pop()
                i += 1
            if not stack:
                boxed_content = expr[start:i-1]
                positions.append(boxed_content)
            idx = i
        else:
            idx += 1
    if positions:
        if(positions[-1]=="self-contradiction"):
            positions[-1]="self_contradiction"  
        if positions[-1].replace("\\", "") == gt[0].replace("\\", ""):
            return 1.0
        else:
            return 0.0
    else:
        return 0.0

def call_build_description(obj, args):
    """
    1. 获取 obj.build_description 方法的参数名
    2. 从 args 里挑出匹配的参数
    3. 调用 obj.build_description 并传入筛选后的参数
    """
    # 获取 `build_description` 方法的参数信息
    method_signature = inspect.signature(obj.build_description)

    # 获取该方法真正需要的参数名
    valid_params = set(method_signature.parameters.keys())

    # 只保留 args 中的匹配参数
    filtered_args = {k: v for k, v in args.items() if k in valid_params}

    # 调用 `build_description` 方法
    return obj.build_description(**filtered_args)


def instruction_compute_score(answer, item):
    res=0
    # pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    # format_match = re.fullmatch(pattern, answer)
    # if format_match:
    #     res+=2.5

  
    # First extract content between <answer> tags
    # answer_pattern = r'<answer>(.*?)</answer>'
    # answer_match = re.search(answer_pattern, answer, re.DOTALL)
    answer_match = re.search(r"</think>\n\n(.*)", answer, re.DOTALL)
    if not answer_match:
        return 0
    
    ids_to_check = item['instruction_id_list']
    args_to_check = item['kwargs']
    constraints=item['constraints']
    resp_to_check = answer_match.group(1)
    #print(resp_to_check)
    


    is_following_list = []
    for ids, arg,con in zip(ids_to_check, args_to_check,constraints):
        if ids in ['soft_content:language', 'soft','soft_content:open_ended', 'situation:suggestion', 'situation:role_play', 'situation:story_generation', 'style:open_ended']:
            url = "http://172.30.44.182:5000/predict"#
            data = {
                "answer": resp_to_check,
                "question": con
            }
            response = requests.post(url,json=data)
            result = response.json()
            res+=result['value']
            continue
        if ids in ['light']:
            res=compute_score(resp_to_check,constraints[0])
            return res*5
        if ids in ['light_choice']:
            boxed_answer = extract_boxed_content(resp_to_check)
            if boxed_answer == "None":
                matches = re.findall(r'boxed{(.*?)}', resp_to_check)
                if matches:
                    boxed_answer = matches[-1]  # 取最后一个匹配的内容
            if boxed_answer == constraints[0]:
                return 5
            else:
                return 0
        instruction_cls = instructions_registry.INSTRUCTION_DICT[ids]
        instruction = instruction_cls(ids)
        call_build_description(instruction, arg)
        
        if resp_to_check.strip() and instruction.check_following(resp_to_check):
            res+=1
        else:
            res+=0
    res=res*(5/len(constraints))
    # Normalize the score to be between 0 and 1
    #score = is_following_list.count(True) / len(is_following_list)
    return res
    

def instruction_val_compute_score(answer, item):

   
    import requests
    import json
    
    
    
    # answer_pattern = r'<answer>(.*?)</answer>'
    # answer_match = re.search(answer_pattern, answer, re.DOTALL)
    answer_match = re.search(r"</think>\n\n(.*)", answer, re.DOTALL)
    if not answer_match:
        # print("not answer_match")
        return 0,0,len(item),0
    
    ids_to_check = item['instruction_id_list']
    args_to_check = item['kwargs']
    resp_to_check = answer_match.group(1)
    
    # print('=====test=====')
    # print(resp_to_check)
    # print('=====test=====')

    is_following_list = []
    for ids, arg in zip(ids_to_check, args_to_check):

        instruction_cls = instructions_registry.INSTRUCTION_DICT[ids]
        instruction = instruction_cls(ids)
        call_build_description(instruction, arg)
        
        if resp_to_check.strip() and instruction.check_following(resp_to_check):
            is_following_list.append(True)
        else:
            is_following_list.append(False)
            
    # Normalize the score to be between 0 and 1
    score = is_following_list.count(True) / len(is_following_list)
    
    score1=score

    all_right=0
    if score == 1:
        all_right=1
    
    return all_right,is_following_list.count(True),len(is_following_list),score1

