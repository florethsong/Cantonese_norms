# -*- coding:utf-8 -*-
'''
@file   :model.py
@author :Floret Huacheng SONG
@time   :20/11/2025 下午2:55
@purpose: for accessing API of target LLMs
'''


from prompt import prompt_selection
import pandas as pd
import os
import time
from tqdm import tqdm
import json
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# 读取匹配数据，生成字典
def data_generation(inpath, language):
    df = pd.read_excel(inpath)

    if language == 'simple':
        char_col = 'S_Character'
        gold_AoA_col = 'S_AoA'
        gold_Fam_col = 'S_Familiarity'
        gold_Conc_col = 'S_Concreteness'
        gold_Image_col = 'S_Imageability'

    elif language == 'traditional':
        char_col = 'T_Character'
        gold_AoA_col = 'T_AoA'
        gold_Fam_col = 'T_Familiarity'
        gold_Conc_col = 'T_Concreteness'
        gold_Image_col = 'T_Imageability'

    else:
        raise ValueError("language must be 'simple' or 'traditional'")

    result_dict = {}

    for _, row in df.iterrows():
        key = row[char_col]

        result_dict[key] = {
            "gold_AoA": row[gold_AoA_col],
            "gold_Familiarity": row[gold_Fam_col],
            "gold_Concreteness": row[gold_Conc_col],
            "gold_Imageability": row[gold_Image_col]
        }

    return result_dict

# Qwen2.5-72b-instruct
def qwen_bailian_2(type, task, mode, input, max_retries=3, timeout=10):
    client = OpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    prompt_1, prompt_2 = prompt_selection(type, task, mode, input)
    print(prompt_1)
    print(prompt_2)

    messages = [
        {"role": "system", "content": prompt_1},
        {"role": "user", "content": prompt_2},
    ]

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                response_format={"type": "json_object"},
                messages=messages,
                # stream=True,
                timeout=timeout,
                temperature=0,
                top_p=1,
                extra_body={"enable_thinking": False}
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                # 所有重试都失败，返回默认结果
                default_result = {"Character": 'NaN', task.capitalize(): 'NaN'}
                return default_result
            time.sleep(5)  # 等待5秒后重试

# DeepSeek-v3
def deepseek_bailian_2(type, task, mode, input, max_retries=3, timeout=10):
    client = OpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    prompt_1, prompt_2 = prompt_selection(type, task, mode, input)
    print(prompt_1)
    print(prompt_2)

    messages = [
        {"role": "system", "content": prompt_1},
        {"role": "user", "content": prompt_2},
    ]

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="deepseek-v3",
                response_format={"type": "json_object"},
                messages=messages,
                # stream=True,
                timeout=timeout,
                temperature=0,
                top_p=1,
                extra_body={"enable_thinking": False}
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                # 所有重试都失败，返回默认结果
                default_result = {"Character": 'NaN', task.capitalize(): 'NaN'}
                return default_result
            time.sleep(5)  # 等待5秒后重试

# gpt-4.1-mini
def gpt_openrouter_2(type, task, mode, input, max_retries=3, timeout=10):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="",
    )

    prompt_1, prompt_2 = prompt_selection(type, task, mode, input)
    print(prompt_1)
    print(prompt_2)

    messages = [
        {"role": "system", "content": prompt_1},
        {"role": "user", "content": prompt_2},
    ]

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-4.1-mini",
                response_format={"type": "json_object"},
                messages=messages,
                # stream=True,
                timeout=timeout,
                temperature=0,
                top_p=1,
                extra_body={"enable_thinking": False, "provider": {"sort": "price"}}
            )

            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                # 所有重试都失败，返回默认结果
                default_result = {"Character": 'NaN', task.capitalize(): 'NaN'}
                return default_result
            time.sleep(5)  # 等待5秒后重试

# Mistral 3.2 24b-instruct
def mistral_openrouter_2(type, task, mode, input, max_retries=3, timeout=10):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="",
    )

    prompt_1, prompt_2 = prompt_selection(type, task, mode, input)
    print(prompt_1)
    print(prompt_2)

    messages = [
        {"role": "system", "content": prompt_1},
        {"role": "user", "content": prompt_2},
    ]

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="mistralai/mistral-small-3.2-24b-instruct",
                response_format={"type": "json_object"},
                messages=messages,
                # stream=True,
                timeout=timeout,
                temperature=0,
                top_p=1,
                extra_body={"enable_thinking": False, "provider": {"sort": "price"}}
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                # 所有重试都失败，返回默认结果
                default_result = {"Character": 'NaN', task.capitalize(): 'NaN'}
                return default_result
            time.sleep(5)  # 等待5秒后重试

# Llama-3-70b-instruct
def llama_openrouter_2(type, task, mode, input, max_retries=3, timeout=10):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="",
    )

    prompt_1, prompt_2 = prompt_selection(type, task, mode, input)
    print(prompt_1)
    print(prompt_2)

    messages = [
        {"role": "system", "content": prompt_1},
        {"role": "user", "content": prompt_2},
    ]

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-3-70b-instruct",
                response_format={"type": "json_object"},
                messages=messages,
                # stream=True,
                timeout=timeout,
                temperature=0,
                top_p=1,
                extra_body={"enable_thinking": False, "provider": {"sort": "price"}}
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                # 所有重试都失败，返回默认结果
                default_result = {"Character": 'NaN', task.capitalize(): 'NaN'}
                return default_result
            time.sleep(5)  # 等待5秒后重试

def process_characters_2(language, type, task, mode, model):
    """
    language: 'simple', 'traditional'
    type: 'en', 'zh', or 'ca'
    task: 'AoA', 'imageability', 'concreteness', 'familiarity'
    mode: 'zero' (zero-shot) or 'few' (few-shot)
    model: 'qwen', 'deepseek', 'llama', 'gpt'
    """

    input_path = r'@merged.xlsx'
    input_dict = data_generation(input_path, language)
    input_list = list(input_dict.keys())

    output_dir= r'round1' #round2 # round3
    output_path=f"{output_dir}\\{language}_{task}_{type}_{mode}_{model}.jsonl"
    all_output_path=f"{output_dir}\\{language}_{task}_{type}_{mode}_{model}.json"

    results = []

    # 创建进度条
    pbar = tqdm(input_list, desc="Processing characters")

    for id, input in enumerate(pbar):
        try:
            # 更新进度条描述
            pbar.set_postfix({"current": input[:20] + "..."})

            # 调用LLM处理
            start_time = time.time()

            if model == 'qwen':
                result = qwen_bailian_2(type, task, mode, input)
            elif model == 'deepseek':
                result = deepseek_bailian_2(type, task, mode, input)
            elif model == 'llama':
                result = llama_openrouter_2(type, task, mode, input)
            elif model == 'gpt':
                result = gpt_openrouter_2(type, task, mode, input)
            elif model == 'mistral':
                result = mistral_openrouter_2(type, task, mode, input)
            else:
                print(f"Invalid mode. Choose 'qwen', 'deepseek', 'llama', 'gpt', 'mistral'.")
                result = {"Character": 'NaN', task.capitalize(): 'NaN'}

            processing_time = time.time() - start_time
            print(processing_time)
            time_path = f"{output_dir}\\times\\{language}_{task}_{type}_{mode}_{model}.jsonl"
            with open(time_path, "a") as f1:
                f1.write(json.dumps(processing_time) + "\n")

            # 验证结果格式
            try:
                # result.lstrip('"').rstrip('"').replece('\\"', '"')
                json.loads(result)  # 验证是否为合法JSON
                results.append(json.loads(result))
                print(f"Generated content: {result}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON output for: {input}:{e}")
                print(f"Generated content: {result}")
                results.append(result)

            # 实时保存每个句子的结果
            with open(output_path, "a") as f2:
                f2.write(json.dumps(result) + "\n")

        except Exception as e:
            result = {"Character": 'NaN', task.capitalize(): 'NaN'}
            print(f"Error processing sentence: {input[:50]}... Error: {str(e)}")
            results.append(result)

    # 保存完整结果
    with open(all_output_path, "w", encoding='utf-8') as f3:
        json.dump(results, f3, ensure_ascii=False, indent=2)

    return results

def incomplete_characters_2(incompath):
    """
    language: 'simple', 'traditional'
    type: 'en', 'zh', or 'ca'
    task: 'AoA', 'imageability', 'concreteness', 'familiarity'
    mode: 'zero' (zero-shot) or 'few' (few-shot)
    model: 'qwen', 'deepseek', 'llama', 'gpt'
    """

    file_name = os.path.splitext(os.path.basename(incompath))[0]
    language, task, type, mode, model = file_name.split('_')
    # print(f'{language}, {task}, {type}, {mode}, {model}')

    #加载已存在的jsonl文件结果
    results = []
    try:
        with open(incompath, 'r', encoding='utf-8') as f1:
            for line in f1:
                print(line)
                if line.strip():
                    try:
                        # 尝试解析为JSON
                        results.append(json.loads(line.strip()))
                    except:
                        # 如果不是有效JSON，直接存储原始字符串
                        results.append(line.strip())
        print(f"已加载 {len(results)} 个现有结果")
    except Exception as e:
        print(f"加载现有结果时出错: {e}")
        results = []

    # 统计已存在的结果
    processed_count = len(results)

    input_path = r'@merged.xlsx'
    input_dict = data_generation(input_path, language)
    input_list = list(input_dict.keys())
    remaining_list = input_list[processed_count:]

    print(f"从索引 {processed_count} 开始处理，剩余 {len(remaining_list)} 个输入")

    output_dir= r'\round1\imcompleted'
    output_path=f"{output_dir}\\{language}_{task}_{type}_{mode}_{model}+.jsonl"
    all_output_path=f"{output_dir}\\{language}_{task}_{type}_{mode}_{model}.json"

    # 创建进度条
    pbar = tqdm(remaining_list, desc="Processing characters")

    new_results = results.copy()  # 从现有结果开始

    for idx, input_item in enumerate(pbar):
        try:
            # 更新进度条描述
            pbar.set_postfix({"current": input_item[:20] + "..."})

            # 调用LLM处理
            start_time = time.time()

            if model == 'qwen':
                result = qwen_bailian_2(type, task, mode, input_item)
            elif model == 'deepseek':
                result = deepseek_bailian_2(type, task, mode, input_item)
            elif model == 'llama':
                result = llama_openrouter_2(type, task, mode, input_item)
            elif model == 'gpt':
                result = gpt_openrouter_2(type, task, mode, input_item)
            elif model == 'mistral':
                result = mistral_openrouter_2(type, task, mode, input_item)
            else:
                print(f"Invalid mode. Choose 'qwen', 'deepseek', 'llama', 'gpt', 'mistral'.")
                result = {"Character": 'NaN', task.capitalize(): 'NaN'}

            processing_time = time.time() - start_time
            print(f"处理时间: {processing_time:.2f}秒")

            # 验证结果格式并存储
            try:
                parsed_result = json.loads(result)
                new_results.append(parsed_result)
                print(f"Generated content: {result}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON output for: {input}:{e}")
                print(f"Generated content: {result}")
                new_results.append(result)

            # 实时追加到jsonl文件（作为字符串）
            with open(output_path, "a") as f2:
                f2.write(json.dumps(result) + "\n")

        except Exception as e:
            error_result = {"Character": 'NaN', task.capitalize(): 'NaN', "error": str(e)}
            print(f"处理错误: {input_item[:50]}... - 错误: {str(e)}")
            new_results.append(error_result)

    # 保存完整结果（覆盖）
    with open(all_output_path, "w", encoding='utf-8') as f3:
        json.dump(new_results, f3, ensure_ascii=False, indent=2)

    print(f"处理完成! 总结果数: {len(new_results)}")
    return new_results

# Qwen2.5-7b-instruct
def qwen_7b(type, task, mode, input, max_retries=3, timeout=5):
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False, # 是否信任远程代码
        revision=None,           # 模型版本，如 "main", "v1.0"
        low_cpu_mem_usage=False, # 减少CPU内存使用
        use_safetensors=True,    # 是否使用safetensors格式
        cache_dir=None,          # 缓存目录
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt_1, prompt_2 = prompt_selection(type, task, mode, input)
    print(prompt_1)
    print(prompt_2)

    messages = [
        {"role": "system", "content": prompt_1},
        {"role": "user", "content": prompt_2},
    ]

    for attempt in range(max_retries):
        try:
            text = tokenizer.apply_chat_template(
                messages,  # 对话消息列表
                tokenize=False,  # 是否返回tokenized结果
                add_generation_prompt=True,  # 是否添加生成提示（如<|im_start|>assistant）
                padding=False,  # 是否填充
                truncation=False,  # 是否截断
                max_length=None,  # 最大长度
                # return_tensors="pt",  # 返回张量格式，"pt"或"tf"
                return_dict=False,  # 是否返回字典
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,  # 输入张量
                max_new_tokens=512,  # 最大生成token数
                temperature=0,  # 温度参数（0=贪婪解码）
                do_sample=False,  # 是否采样（False时为贪婪解码）
                top_k=1,  # top-k采样
                top_p=1.0,  # nucleus采样（top-p）
                repetition_penalty=1.0,  # 重复惩罚
                num_beams=1,  # beam search的beam数
                early_stopping=False,  # 是否提前停止
                no_repeat_ngram_size=0,  # 禁止重复的n-gram大小
                num_return_sequences=1,  # 返回的序列数
                pad_token_id=tokenizer.pad_token_id,  # 填充token ID
                eos_token_id=tokenizer.eos_token_id,  # 结束token ID
                bos_token_id=tokenizer.bos_token_id,  # 开始token ID
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                # 所有重试都失败，返回默认结果
                default_result = {"Character": 'NaN', task.capitalize(): 'NaN'}
                return default_result
            time.sleep(timeout)  # 等待5秒后重试

# CantoneseLLMChat-v1.0-7B
def qwen_7b_cantonese(type, task, mode, input, max_retries=3, timeout=5):
    model_name = "hon9kon9ize/CantoneseLLMChat-v1.0-7B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False, # 是否信任远程代码
        revision=None,           # 模型版本，如 "main", "v1.0"
        low_cpu_mem_usage=False, # 减少CPU内存使用
        use_safetensors=True,    # 是否使用safetensors格式
        cache_dir=None,          # 缓存目录
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt_1, prompt_2 = prompt_selection(type, task, mode, input)
    print(prompt_1)
    print(prompt_2)

    messages = [
        {"role": "system", "content": prompt_1},
        {"role": "user", "content": prompt_2},
    ]

    for attempt in range(max_retries):
        try:
            text = tokenizer.apply_chat_template(
                messages,  # 对话消息列表
                tokenize=False,  # 是否返回tokenized结果
                add_generation_prompt=True,  # 是否添加生成提示（如<|im_start|>assistant）
                padding=False,  # 是否填充
                truncation=False,  # 是否截断
                max_length=None,  # 最大长度
                # return_tensors="pt",  # 返回张量格式，"pt"或"tf"
                return_dict=False,  # 是否返回字典
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,  # 输入张量
                max_new_tokens=512,  # 最大生成token数
                temperature=0,  # 温度参数（0=贪婪解码）
                do_sample=False,  # 是否采样（False时为贪婪解码）
                top_k=1,  # top-k采样
                top_p=1.0,  # nucleus采样（top-p）
                repetition_penalty=1.0,  # 重复惩罚
                num_beams=1,  # beam search的beam数
                early_stopping=False,  # 是否提前停止
                no_repeat_ngram_size=0,  # 禁止重复的n-gram大小
                num_return_sequences=1,  # 返回的序列数
                pad_token_id=tokenizer.pad_token_id,  # 填充token ID
                eos_token_id=tokenizer.eos_token_id,  # 结束token ID
                bos_token_id=tokenizer.bos_token_id,  # 开始token ID
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                # 所有重试都失败，返回默认结果
                default_result = {"Character": 'NaN', task.capitalize(): 'NaN'}
                return default_result
            time.sleep(timeout)  # 等待5秒后重试

def process_characters_3(language, type, task, mode, model):
    """
    language: 'simple', 'traditional'
    type: 'en', 'zh', or 'ca'
    task: 'AoA', 'imageability', 'concreteness', 'familiarity'
    mode: 'zero' (zero-shot) or 'few' (few-shot)
    model: 'qwen7b', 'qwen7bc'
    """

    input_path = r"@merged.xlsx"
    input_dict = data_generation(input_path, language)
    input_list = list(input_dict.keys())

    output_dir= r"round3" #round2 # round3
    output_path=f"{output_dir}/{language}_{task}_{type}_{mode}_{model}.jsonl"
    all_output_path=f"{output_dir}/{language}_{task}_{type}_{mode}_{model}.json"

    results = []

    # 创建进度条
    pbar = tqdm(input_list, desc="Processing characters")

    for id, input in enumerate(pbar):
        try:
            # 更新进度条描述
            pbar.set_postfix({"current": input[:20] + "..."})

            # 调用LLM处理
            start_time = time.time()

            if model == 'qwen7b':
                result = qwen_7b(type, task, mode, input)
            elif model == 'qwen7bc':
                result = qwen_7b_cantonese(type, task, mode, input)
            else:
                print(f"Invalid mode. Choose 'qwen7b', 'qwen7bc'.")
                result = {"Character": 'NaN', task.capitalize(): 'NaN'}

            processing_time = time.time() - start_time
            print(processing_time)
            time_path = f"{output_dir}/times/{language}_{task}_{type}_{mode}_{model}.jsonl"
            with open(time_path, "a") as f1:
                f1.write(json.dumps(processing_time) + "\n")

            # 验证结果格式
            try:
                # result.lstrip('"').rstrip('"').replece('\\"', '"')
                json.loads(result)  # 验证是否为合法JSON
                results.append(json.loads(result))
                print(f"Generated content: {result}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON output for: {input}:{e}")
                print(f"Generated content: {result}")
                results.append(result)

            # 实时保存每个句子的结果
            with open(output_path, "a") as f2:
                f2.write(json.dumps(result) + "\n")

        except Exception as e:
            result = {"Character": 'NaN', task.capitalize(): 'NaN'}
            print(f"Error processing sentence: {input[:50]}... Error: {str(e)}")
            results.append(result)

    # 保存完整结果
    with open(all_output_path, "w", encoding='utf-8') as f3:
        json.dump(results, f3, ensure_ascii=False, indent=2)

    return results

def incomplete_characters_3(incompath):
    """
    language: 'simple', 'traditional'
    type: 'en', 'zh', or 'ca'
    task: 'AoA', 'imageability', 'concreteness', 'familiarity'
    mode: 'zero' (zero-shot) or 'few' (few-shot)
    model: 'qwen', 'deepseek', 'llama', 'gpt'
    """

    file_name = os.path.splitext(os.path.basename(incompath))[0]
    language, task, type, mode, model = file_name.split('_')
    # print(f'{language}, {task}, {type}, {mode}, {model}')

    #加载已存在的jsonl文件结果
    results = []
    try:
        with open(incompath, 'r', encoding='utf-8') as f1:
            for line in f1:
                print(line)
                if line.strip():
                    try:
                        # 尝试解析为JSON
                        results.append(json.loads(line.strip()))
                    except:
                        # 如果不是有效JSON，直接存储原始字符串
                        results.append(line.strip())
        print(f"已加载 {len(results)} 个现有结果")
    except Exception as e:
        print(f"加载现有结果时出错: {e}")
        results = []

    # 统计已存在的结果
    processed_count = len(results)

    # input_path = r'C:\Users\Floret\Desktop\Cantonese Norms\data\@2\@test.xlsx'
    input_path = r"@merged.xlsx"
    input_dict = data_generation(input_path, language)
    input_list = list(input_dict.keys())
    remaining_list = input_list[processed_count:]

    print(f"从索引 {processed_count} 开始处理，剩余 {len(remaining_list)} 个输入")

    output_dir= r"round3"
    output_path=f"{output_dir}\\{language}_{task}_{type}_{mode}_{model}+.jsonl"
    all_output_path=f"{output_dir}\\{language}_{task}_{type}_{mode}_{model}.json"

    # 创建进度条
    pbar = tqdm(remaining_list, desc="Processing characters")

    new_results = results.copy()  # 从现有结果开始

    for idx, input_item in enumerate(pbar):
        try:
            # 更新进度条描述
            pbar.set_postfix({"current": input_item[:20] + "..."})

            # 调用LLM处理
            start_time = time.time()

            if model == 'qwen7b':
                result = qwen_7b(type, task, mode, input_item)
            elif model == 'qwen7bc':
                result = qwen_7b_cantonese(type, task, mode, input_item)
            else:
                print(f"Invalid mode. Choose 'qwen7b', 'qwen7bc'.")
                result = {"Character": 'NaN', task.capitalize(): 'NaN'}

            processing_time = time.time() - start_time
            print(f"处理时间: {processing_time:.2f}秒")

            # 验证结果格式并存储
            try:
                parsed_result = json.loads(result)
                new_results.append(parsed_result)
                print(f"Generated content: {result}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON output for: {input}:{e}")
                print(f"Generated content: {result}")
                new_results.append(result)

            # 实时追加到jsonl文件（作为字符串）
            with open(output_path, "a") as f2:
                f2.write(json.dumps(result) + "\n")

        except Exception as e:
            error_result = {"Character": 'NaN', task.capitalize(): 'NaN', "error": str(e)}
            print(f"处理错误: {input_item[:50]}... - 错误: {str(e)}")
            new_results.append(error_result)

    # 保存完整结果（覆盖）
    with open(all_output_path, "w", encoding='utf-8') as f3:
        json.dump(new_results, f3, ensure_ascii=False, indent=2)

    print(f"处理完成! 总结果数: {len(new_results)}")
    return new_results

if __name__ == '__main__':
    pass
    setting = [
        # 1: **_en_***_zero_qwen
        ['traditional', 'en', 'AoA', 'zero', 'qwen'],
        ['traditional', 'en', 'imageability', 'zero', 'qwen'],
        ['traditional', 'en', 'concreteness', 'zero', 'qwen'],
        ['traditional', 'en', 'familiarity', 'zero', 'qwen'],
        ['simple', 'en', 'AoA', 'zero', 'qwen'],
        ['simple', 'en', 'imageability', 'zero', 'qwen'],
        ['simple', 'en', 'concreteness', 'zero', 'qwen'],
        ['simple', 'en', 'familiarity', 'zero', 'qwen'],
        # 2: **_zh_***_zero_qwen
        ['traditional', 'zh', 'AoA', 'zero', 'qwen'],
        ['traditional', 'zh', 'imageability', 'zero', 'qwen'],
        ['traditional', 'zh', 'concreteness', 'zero', 'qwen'],
        ['traditional', 'zh', 'familiarity', 'zero', 'qwen'],
        ['simple', 'zh', 'AoA', 'zero', 'qwen'],
        ['simple', 'zh', 'imageability', 'zero', 'qwen'],
        ['simple', 'zh', 'concreteness', 'zero', 'qwen'],
        ['simple', 'zh', 'familiarity', 'zero', 'qwen'],
        # 3: **_ca_***_zero_qwen
        ['traditional', 'ca', 'AoA', 'zero', 'qwen'],
        ['traditional', 'ca', 'imageability', 'zero', 'qwen'],
        ['traditional', 'ca', 'concreteness', 'zero', 'qwen'],
        ['traditional', 'ca', 'familiarity', 'zero', 'qwen'],
        ['simple', 'ca', 'AoA', 'zero', 'qwen'],
        ['simple', 'ca', 'imageability', 'zero', 'qwen'],
        ['simple', 'ca', 'concreteness', 'zero', 'qwen'],
        ['simple', 'ca', 'familiarity', 'zero', 'qwen'],
        # 4: **_en_***_zero_deepseek
        ['traditional', 'en', 'AoA', 'zero', 'deepseek'],
        ['traditional', 'en', 'imageability', 'zero', 'deepseek'],
        ['traditional', 'en', 'concreteness', 'zero', 'deepseek'],
        ['traditional', 'en', 'familiarity', 'zero', 'deepseek'],
        ['simple', 'en', 'AoA', 'zero', 'deepseek'],
        ['simple', 'en', 'imageability', 'zero', 'deepseek'],
        ['simple', 'en', 'concreteness', 'zero', 'deepseek'],
        ['simple', 'en', 'familiarity', 'zero', 'deepseek'],
        # 5: **_zh_***_zero_deepseek
        ['traditional', 'zh', 'AoA', 'zero', 'deepseek'],
        ['traditional', 'zh', 'imageability', 'zero', 'deepseek'],
        ['traditional', 'zh', 'concreteness', 'zero', 'deepseek'],
        ['traditional', 'zh', 'familiarity', 'zero', 'deepseek'],
        ['simple', 'zh', 'AoA', 'zero', 'deepseek'],
        ['simple', 'zh', 'imageability', 'zero', 'deepseek'],
        ['simple', 'zh', 'concreteness', 'zero', 'deepseek'],
        ['simple', 'zh', 'familiarity', 'zero', 'deepseek'],
        # 6: **_ca_***_zero_deepseek
        ['traditional', 'ca', 'AoA', 'zero', 'deepseek'],
        ['traditional', 'ca', 'imageability', 'zero', 'deepseek'],
        ['traditional', 'ca', 'concreteness', 'zero', 'deepseek'],
        ['traditional', 'ca', 'familiarity', 'zero', 'deepseek'],
        ['simple', 'ca', 'AoA', 'zero', 'deepseek'],
        ['simple', 'ca', 'imageability', 'zero', 'deepseek'],
        ['simple', 'ca', 'concreteness', 'zero', 'deepseek'],
        ['simple', 'ca', 'familiarity', 'zero', 'deepseek'],
        # 7: **_en_***_zero_llama
        ['traditional', 'en', 'AoA', 'zero', 'llama'],
        ['traditional', 'en', 'imageability', 'zero', 'llama'],
        ['traditional', 'en', 'concreteness', 'zero', 'llama'],
        ['traditional', 'en', 'familiarity', 'zero', 'llama'],
        ['simple', 'en', 'AoA', 'zero', 'llama'],
        ['simple', 'en', 'imageability', 'zero', 'llama'],
        ['simple', 'en', 'concreteness', 'zero', 'llama'],
        ['simple', 'en', 'familiarity', 'zero', 'llama'],
        # 8: **_zh_***_zero_llama
        ['traditional', 'zh', 'AoA', 'zero', 'llama'],
        ['traditional', 'zh', 'imageability', 'zero', 'llama'],
        ['traditional', 'zh', 'concreteness', 'zero', 'llama'],
        ['traditional', 'zh', 'familiarity', 'zero', 'llama'],
        ['simple', 'zh', 'AoA', 'zero', 'llama'],
        ['simple', 'zh', 'imageability', 'zero', 'llama'],
        ['simple', 'zh', 'concreteness', 'zero', 'llama'],
        ['simple', 'zh', 'familiarity', 'zero', 'llama'],
        # 9: **_ca_***_zero_llama
        ['traditional', 'ca', 'AoA', 'zero', 'llama'],
        ['traditional', 'ca', 'imageability', 'zero', 'llama'],
        ['traditional', 'ca', 'concreteness', 'zero', 'llama'],
        ['traditional', 'ca', 'familiarity', 'zero', 'llama'],
        ['simple', 'ca', 'AoA', 'zero', 'llama'],
        ['simple', 'ca', 'imageability', 'zero', 'llama'],
        ['simple', 'ca', 'concreteness', 'zero', 'llama'],
        ['simple', 'ca', 'familiarity', 'zero', 'llama'],
        # 10: **_en_***_zero_gpt
        ['traditional', 'en', 'AoA', 'zero', 'gpt'],
        ['traditional', 'en', 'imageability', 'zero', 'gpt'],
        ['traditional', 'en', 'concreteness', 'zero', 'gpt'],
        ['traditional', 'en', 'familiarity', 'zero', 'gpt'],
        ['simple', 'en', 'AoA', 'zero', 'gpt'],
        ['simple', 'en', 'imageability', 'zero', 'gpt'],
        ['simple', 'en', 'concreteness', 'zero', 'gpt'],
        ['simple', 'en', 'familiarity', 'zero', 'gpt'],
        # 11: **_zh_***_zero_gpt
        ['traditional', 'zh', 'AoA', 'zero', 'gpt'],
        ['traditional', 'zh', 'imageability', 'zero', 'gpt'],
        ['traditional', 'zh', 'concreteness', 'zero', 'gpt'],
        ['traditional', 'zh', 'familiarity', 'zero', 'gpt'],
        ['simple', 'zh', 'AoA', 'zero', 'gpt'],
        ['simple', 'zh', 'imageability', 'zero', 'gpt'],
        ['simple', 'zh', 'concreteness', 'zero', 'gpt'],
        ['simple', 'zh', 'familiarity', 'zero', 'gpt'],
        # 12: **_ca_***_zero_gpt
        ['traditional', 'ca', 'AoA', 'zero', 'gpt'],
        ['traditional', 'ca', 'imageability', 'zero', 'gpt'],
        ['traditional', 'ca', 'concreteness', 'zero', 'gpt'],
        ['traditional', 'ca', 'familiarity', 'zero', 'gpt'],
        ['simple', 'ca', 'AoA', 'zero', 'gpt'],
        ['simple', 'ca', 'imageability', 'zero', 'gpt'],
        ['simple', 'ca', 'concreteness', 'zero', 'gpt'],
        ['simple', 'ca', 'familiarity', 'zero', 'gpt'],
        # 13: **_en_***_few_qwen
        ['traditional', 'en', 'AoA', 'few', 'qwen'],
        ['traditional', 'en', 'imageability', 'few', 'qwen'],
        ['traditional', 'en', 'concreteness', 'few', 'qwen'],
        ['traditional', 'en', 'familiarity', 'few', 'qwen'],
        ['simple', 'en', 'AoA', 'few', 'qwen'],
        ['simple', 'en', 'imageability', 'few', 'qwen'],
        ['simple', 'en', 'concreteness', 'few', 'qwen'],
        ['simple', 'en', 'familiarity', 'few', 'qwen'],
        # 14: **_zh_***_few_qwen
        ['traditional', 'zh', 'AoA', 'few', 'qwen'],
        ['traditional', 'zh', 'imageability', 'few', 'qwen'],
        ['traditional', 'zh', 'concreteness', 'few', 'qwen'],
        ['traditional', 'zh', 'familiarity', 'few', 'qwen'],
        ['simple', 'zh', 'AoA', 'few', 'qwen'],
        ['simple', 'zh', 'imageability', 'few', 'qwen'],
        ['simple', 'zh', 'concreteness', 'few', 'qwen'],
        ['simple', 'zh', 'familiarity', 'few', 'qwen'],
        # 15: **_ca_***_few_qwen
        ['traditional', 'ca', 'AoA', 'few', 'qwen'],
        ['traditional', 'ca', 'imageability', 'few', 'qwen'],
        ['traditional', 'ca', 'concreteness', 'few', 'qwen'],
        ['traditional', 'ca', 'familiarity', 'few', 'qwen'],
        ['simple', 'ca', 'AoA', 'few', 'qwen'],
        ['simple', 'ca', 'imageability', 'few', 'qwen'],
        ['simple', 'ca', 'concreteness', 'few', 'qwen'],
        ['simple', 'ca', 'familiarity', 'few', 'qwen'],
        # 16: **_en_***_few_deepseek
        ['traditional', 'en', 'AoA', 'few', 'deepseek'],
        ['traditional', 'en', 'imageability', 'few', 'deepseek'],
        ['traditional', 'en', 'concreteness', 'few', 'deepseek'],
        ['traditional', 'en', 'familiarity', 'few', 'deepseek'],
        ['simple', 'en', 'AoA', 'few', 'deepseek'],
        ['simple', 'en', 'imageability', 'few', 'deepseek'],
        ['simple', 'en', 'concreteness', 'few', 'deepseek'],
        ['simple', 'en', 'familiarity', 'few', 'deepseek'],
        # 17: **_zh_***_few_deepseek
        ['traditional', 'zh', 'AoA', 'few', 'deepseek'],
        ['traditional', 'zh', 'imageability', 'few', 'deepseek'],
        ['traditional', 'zh', 'concreteness', 'few', 'deepseek'],
        ['traditional', 'zh', 'familiarity', 'few', 'deepseek'],
        ['simple', 'zh', 'AoA', 'few', 'deepseek'],
        ['simple', 'zh', 'imageability', 'few', 'deepseek'],
        ['simple', 'zh', 'concreteness', 'few', 'deepseek'],
        ['simple', 'zh', 'familiarity', 'few', 'deepseek'],
        # 18: **_ca_***_few_deepseek
        ['traditional', 'ca', 'AoA', 'few', 'deepseek'],
        ['traditional', 'ca', 'imageability', 'few', 'deepseek'],
        ['traditional', 'ca', 'concreteness', 'few', 'deepseek'],
        ['traditional', 'ca', 'familiarity', 'few', 'deepseek'],
        ['simple', 'ca', 'AoA', 'few', 'deepseek'],
        ['simple', 'ca', 'imageability', 'few', 'deepseek'],
        ['simple', 'ca', 'concreteness', 'few', 'deepseek'],
        ['simple', 'ca', 'familiarity', 'few', 'deepseek'],
        # 19: **_en_***_few_llama
        ['traditional', 'en', 'AoA', 'few', 'llama'],
        ['traditional', 'en', 'imageability', 'few', 'llama'],
        ['traditional', 'en', 'concreteness', 'few', 'llama'],
        ['traditional', 'en', 'familiarity', 'few', 'llama'],
        ['simple', 'en', 'AoA', 'few', 'llama'],
        ['simple', 'en', 'imageability', 'few', 'llama'],
        ['simple', 'en', 'concreteness', 'few', 'llama'],
        ['simple', 'en', 'familiarity', 'few', 'llama'],
        # 20: **_zh_***_few_llama
        ['traditional', 'zh', 'AoA', 'few', 'llama'],
        ['traditional', 'zh', 'imageability', 'few', 'llama'],
        ['traditional', 'zh', 'concreteness', 'few', 'llama'],
        ['traditional', 'zh', 'familiarity', 'few', 'llama'],
        ['simple', 'zh', 'AoA', 'few', 'llama'],
        ['simple', 'zh', 'imageability', 'few', 'llama'],
        ['simple', 'zh', 'concreteness', 'few', 'llama'],
        ['simple', 'zh', 'familiarity', 'few', 'llama'],
        # 21: **_ca_***_few_llama
        ['traditional', 'ca', 'AoA', 'few', 'llama'],
        ['traditional', 'ca', 'imageability', 'few', 'llama'],
        ['traditional', 'ca', 'concreteness', 'few', 'llama'],
        ['traditional', 'ca', 'familiarity', 'few', 'llama'],
        ['simple', 'ca', 'AoA', 'few', 'llama'],
        ['simple', 'ca', 'imageability', 'few', 'llama'],
        ['simple', 'ca', 'concreteness', 'few', 'llama'],
        ['simple', 'ca', 'familiarity', 'few', 'llama'],
        # 22: **_en_***_few_gpt
        ['traditional', 'en', 'AoA', 'few', 'gpt'],
        ['traditional', 'en', 'imageability', 'few', 'gpt'],
        ['traditional', 'en', 'concreteness', 'few', 'gpt'],
        ['traditional', 'en', 'familiarity', 'few', 'gpt'],
        ['simple', 'en', 'AoA', 'few', 'gpt'],
        ['simple', 'en', 'imageability', 'few', 'gpt'],
        ['simple', 'en', 'concreteness', 'few', 'gpt'],
        ['simple', 'en', 'familiarity', 'few', 'gpt'],
        # 23: **_zh_***_few_gpt
        ['traditional', 'zh', 'AoA', 'few', 'gpt'],
        ['traditional', 'zh', 'imageability', 'few', 'gpt'],
        ['traditional', 'zh', 'concreteness', 'few', 'gpt'],
        ['traditional', 'zh', 'familiarity', 'few', 'gpt'],
        ['simple', 'zh', 'AoA', 'few', 'gpt'],
        ['simple', 'zh', 'imageability', 'few', 'gpt'],
        ['simple', 'zh', 'concreteness', 'few', 'gpt'],
        ['simple', 'zh', 'familiarity', 'few', 'gpt'],
        # 24: **_ca_***_few_gpt
        ['traditional', 'ca', 'AoA', 'few', 'gpt'],
        ['traditional', 'ca', 'imageability', 'few', 'gpt'],
        ['traditional', 'ca', 'concreteness', 'few', 'gpt'],
        ['traditional', 'ca', 'familiarity', 'few', 'gpt'],
        ['simple', 'ca', 'AoA', 'few', 'gpt'],
        ['simple', 'ca', 'imageability', 'few', 'gpt'],
        ['simple', 'ca', 'concreteness', 'few', 'gpt'],
        ['simple', 'ca', 'familiarity', 'few', 'gpt'],

        # + mistral
        # 25: **_en_***_zero_mistral
        ['traditional', 'en', 'AoA', 'zero', 'mistral'],
        ['traditional', 'en', 'imageability', 'zero', 'mistral'],
        ['traditional', 'en', 'concreteness', 'zero', 'mistral'],
        ['traditional', 'en', 'familiarity', 'zero', 'mistral'],
        ['simple', 'en', 'AoA', 'zero', 'mistral'],
        ['simple', 'en', 'imageability', 'zero', 'mistral'],
        ['simple', 'en', 'concreteness', 'zero', 'mistral'],
        ['simple', 'en', 'familiarity', 'zero', 'mistral'],
        # 26: **_zh_***_zero_mistral
        ['traditional', 'zh', 'AoA', 'zero', 'mistral'],
        ['traditional', 'zh', 'imageability', 'zero', 'mistral'],
        ['traditional', 'zh', 'concreteness', 'zero', 'mistral'],
        ['traditional', 'zh', 'familiarity', 'zero', 'mistral'],
        ['simple', 'zh', 'AoA', 'zero', 'mistral'],
        ['simple', 'zh', 'imageability', 'zero', 'mistral'],
        ['simple', 'zh', 'concreteness', 'zero', 'mistral'],
        ['simple', 'zh', 'familiarity', 'zero', 'mistral'],
        # 27: **_ca_***_zero_mistral
        ['traditional', 'ca', 'AoA', 'zero', 'mistral'],
        ['traditional', 'ca', 'imageability', 'zero', 'mistral'],
        ['traditional', 'ca', 'concreteness', 'zero', 'mistral'],
        ['traditional', 'ca', 'familiarity', 'zero', 'mistral'],
        ['simple', 'ca', 'AoA', 'zero', 'mistral'],
        ['simple', 'ca', 'imageability', 'zero', 'mistral'],
        ['simple', 'ca', 'concreteness', 'zero', 'mistral'],
        ['simple', 'ca', 'familiarity', 'zero', 'mistral'],
        # 28: **_en_***_few_mistral
        ['traditional', 'en', 'AoA', 'few', 'mistral'],
        ['traditional', 'en', 'imageability', 'few', 'mistral'],
        ['traditional', 'en', 'concreteness', 'few', 'mistral'],
        ['traditional', 'en', 'familiarity', 'few', 'mistral'],
        ['simple', 'en', 'AoA', 'few', 'mistral'],
        ['simple', 'en', 'imageability', 'few', 'mistral'],
        ['simple', 'en', 'concreteness', 'few', 'mistral'],
        ['simple', 'en', 'familiarity', 'few', 'mistral'],
        # 29: **_zh_***_few_mistral
        ['traditional', 'zh', 'AoA', 'few', 'mistral'],
        ['traditional', 'zh', 'imageability', 'few', 'mistral'],
        ['traditional', 'zh', 'concreteness', 'few', 'mistral'],
        ['traditional', 'zh', 'familiarity', 'few', 'mistral'],
        ['simple', 'zh', 'AoA', 'few', 'mistral'],
        ['simple', 'zh', 'imageability', 'few', 'mistral'],
        ['simple', 'zh', 'concreteness', 'few', 'mistral'],
        ['simple', 'zh', 'familiarity', 'few', 'mistral'],
        # 30: **_ca_***_few_mistral
        ['traditional', 'ca', 'AoA', 'few', 'mistral'],
        ['traditional', 'ca', 'imageability', 'few', 'mistral'],
        ['traditional', 'ca', 'concreteness', 'few', 'mistral'],
        ['traditional', 'ca', 'familiarity', 'few', 'mistral'],
        ['simple', 'ca', 'AoA', 'few', 'mistral'],
        ['simple', 'ca', 'imageability', 'few', 'mistral'],
        ['simple', 'ca', 'concreteness', 'few', 'mistral'],
        ['simple', 'ca', 'familiarity', 'few', 'mistral'],

        # + qwen7b
        # 31: **_en_***_zero_qwen7b
        ['traditional', 'en', 'AoA', 'zero', 'qwen7b'],
        ['traditional', 'en', 'imageability', 'zero', 'qwen7b'],
        ['traditional', 'en', 'concreteness', 'zero', 'qwen7b'],
        ['traditional', 'en', 'familiarity', 'zero', 'qwen7b'],
        ['simple', 'en', 'AoA', 'zero', 'qwen7b'],
        ['simple', 'en', 'imageability', 'zero', 'qwen7b'],
        ['simple', 'en', 'concreteness', 'zero', 'qwen7b'],
        ['simple', 'en', 'familiarity', 'zero', 'qwen7b'],
        # 32: **_zh_***_zero_qwen7b
        ['traditional', 'zh', 'AoA', 'zero', 'qwen7b'],
        ['traditional', 'zh', 'imageability', 'zero', 'qwen7b'],
        ['traditional', 'zh', 'concreteness', 'zero', 'qwen7b'],
        ['traditional', 'zh', 'familiarity', 'zero', 'qwen7b'],
        ['simple', 'zh', 'AoA', 'zero', 'qwen7b'],
        ['simple', 'zh', 'imageability', 'zero', 'qwen7b'],
        ['simple', 'zh', 'concreteness', 'zero', 'qwen7b'],
        ['simple', 'zh', 'familiarity', 'zero', 'qwen7b'],
        # 33: **_ca_***_zero_qwen7b
        ['traditional', 'ca', 'AoA', 'zero', 'qwen7b'],
        ['traditional', 'ca', 'imageability', 'zero', 'qwen7b'],
        ['traditional', 'ca', 'concreteness', 'zero', 'qwen7b'],
        ['traditional', 'ca', 'familiarity', 'zero', 'qwen7b'],
        ['simple', 'ca', 'AoA', 'zero', 'qwen7b'],
        ['simple', 'ca', 'imageability', 'zero', 'qwen7b'],
        ['simple', 'ca', 'concreteness', 'zero', 'qwen7b'],
        ['simple', 'ca', 'familiarity', 'zero', 'qwen7b'],
        # 34: **_en_***_few_qwen7b
        ['traditional', 'en', 'AoA', 'few', 'qwen7b'],
        ['traditional', 'en', 'imageability', 'few', 'qwen7b'],
        ['traditional', 'en', 'concreteness', 'few', 'qwen7b'],
        ['traditional', 'en', 'familiarity', 'few', 'qwen7b'],
        ['simple', 'en', 'AoA', 'few', 'qwen7b'],
        ['simple', 'en', 'imageability', 'few', 'qwen7b'],
        ['simple', 'en', 'concreteness', 'few', 'qwen7b'],
        ['simple', 'en', 'familiarity', 'few', 'qwen7b'],
        # 35: **_zh_***_few_qwen7b
        ['traditional', 'zh', 'AoA', 'few', 'qwen7b'],
        ['traditional', 'zh', 'imageability', 'few', 'qwen7b'],
        ['traditional', 'zh', 'concreteness', 'few', 'qwen7b'],
        ['traditional', 'zh', 'familiarity', 'few', 'qwen7b'],
        ['simple', 'zh', 'AoA', 'few', 'qwen7b'],
        ['simple', 'zh', 'imageability', 'few', 'qwen7b'],
        ['simple', 'zh', 'concreteness', 'few', 'qwen7b'],
        ['simple', 'zh', 'familiarity', 'few', 'qwen7b'],
        # 36: **_ca_***_few_qwen7b
        ['traditional', 'ca', 'AoA', 'few', 'qwen7b'],
        ['traditional', 'ca', 'imageability', 'few', 'qwen7b'],
        ['traditional', 'ca', 'concreteness', 'few', 'qwen7b'],
        ['traditional', 'ca', 'familiarity', 'few', 'qwen7b'],
        ['simple', 'ca', 'AoA', 'few', 'qwen7b'],
        ['simple', 'ca', 'imageability', 'few', 'qwen7b'],
        ['simple', 'ca', 'concreteness', 'few', 'qwen7b'],
        ['simple', 'ca', 'familiarity', 'few', 'qwen7b'],
        # + qwen7bc
        # 37: **_en_***_zero_qwen7bc
        ['traditional', 'en', 'AoA', 'zero', 'qwen7bc'],
        ['traditional', 'en', 'imageability', 'zero', 'qwen7bc'],
        ['traditional', 'en', 'concreteness', 'zero', 'qwen7bc'],
        ['traditional', 'en', 'familiarity', 'zero', 'qwen7bc'],
        ['simple', 'en', 'AoA', 'zero', 'qwen7bc'],
        ['simple', 'en', 'imageability', 'zero', 'qwen7bc'],
        ['simple', 'en', 'concreteness', 'zero', 'qwen7bc'],
        ['simple', 'en', 'familiarity', 'zero', 'qwen7bc'],
        # 38: **_zh_***_zero_qwen7bc
        ['traditional', 'zh', 'AoA', 'zero', 'qwen7bc'],
        ['traditional', 'zh', 'imageability', 'zero', 'qwen7bc'],
        ['traditional', 'zh', 'concreteness', 'zero', 'qwen7bc'],
        ['traditional', 'zh', 'familiarity', 'zero', 'qwen7bc'],
        ['simple', 'zh', 'AoA', 'zero', 'qwen7bc'],
        ['simple', 'zh', 'imageability', 'zero', 'qwen7bc'],
        ['simple', 'zh', 'concreteness', 'zero', 'qwen7bc'],
        ['simple', 'zh', 'familiarity', 'zero', 'qwen7bc'],
        # 39: **_ca_***_zero_qwen7bc
        ['traditional', 'ca', 'AoA', 'zero', 'qwen7bc'],
        ['traditional', 'ca', 'imageability', 'zero', 'qwen7bc'],
        ['traditional', 'ca', 'concreteness', 'zero', 'qwen7bc'],
        ['traditional', 'ca', 'familiarity', 'zero', 'qwen7bc'],
        ['simple', 'ca', 'AoA', 'zero', 'qwen7bc'],
        ['simple', 'ca', 'imageability', 'zero', 'qwen7bc'],
        ['simple', 'ca', 'concreteness', 'zero', 'qwen7bc'],
        ['simple', 'ca', 'familiarity', 'zero', 'qwen7bc'],
        # 40: **_en_***_few_qwen7bc
        ['traditional', 'en', 'AoA', 'few', 'qwen7bc'],
        ['traditional', 'en', 'imageability', 'few', 'qwen7bc'],
        ['traditional', 'en', 'concreteness', 'few', 'qwen7bc'],
        ['traditional', 'en', 'familiarity', 'few', 'qwen7bc'],
        ['simple', 'en', 'AoA', 'few', 'qwen7bc'],
        ['simple', 'en', 'imageability', 'few', 'qwen7bc'],
        ['simple', 'en', 'concreteness', 'few', 'qwen7bc'],
        ['simple', 'en', 'familiarity', 'few', 'qwen7bc'],
        # 41: **_zh_***_few_qwen7bc
        ['traditional', 'zh', 'AoA', 'few', 'qwen7bc'],
        ['traditional', 'zh', 'imageability', 'few', 'qwen7bc'],
        ['traditional', 'zh', 'concreteness', 'few', 'qwen7bc'],
        ['traditional', 'zh', 'familiarity', 'few', 'qwen7bc'],
        ['simple', 'zh', 'AoA', 'few', 'qwen7bc'],
        ['simple', 'zh', 'imageability', 'few', 'qwen7bc'],
        ['simple', 'zh', 'concreteness', 'few', 'qwen7bc'],
        ['simple', 'zh', 'familiarity', 'few', 'qwen7bc'],
        # 42: **_ca_***_few_qwen7bc
        ['traditional', 'ca', 'AoA', 'few', 'qwen7bc'],
        ['traditional', 'ca', 'imageability', 'few', 'qwen7bc'],
        ['traditional', 'ca', 'concreteness', 'few', 'qwen7bc'],
        ['traditional', 'ca', 'familiarity', 'few', 'qwen7bc'],
        ['simple', 'ca', 'AoA', 'few', 'qwen7bc'],
        ['simple', 'ca', 'imageability', 'few', 'qwen7bc'],
        ['simple', 'ca', 'concreteness', 'few', 'qwen7bc'],
        ['simple', 'ca', 'familiarity', 'few', 'qwen7bc'],
    ]
