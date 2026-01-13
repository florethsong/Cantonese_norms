# -*- coding:utf-8 -*-
'''
@file   :data.py
@author :Floret Huacheng SONG
@time   :18/11/2025 下午5:19
@purpose: for clearing and adjusting data format and content
'''

import numpy as np
import pandas as pd
from opencc import OpenCC
import os
import glob
import json
import re
import csv
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from model import data_generation
from model import qwen_bailian_2, deepseek_bailian_2, llama_openrouter_2, gpt_openrouter_2, mistral_openrouter_2, qwen_7b, qwen_7b_cantonese
from statistic import MergedDataParser

# 读取两个原始Excel文件, 输出匹配数据
def process_raw_data():
    # 读取简体中文数据
    simplified_df = pd.read_excel(r'Simplified_norm_2007.xlsx')
    traditional_df = pd.read_excel(r'Traditional_norm_2021.xlsx')

    # 提取需要的列
    simplified_data = simplified_df[['Word', 'AoA', 'FAM', 'CON', 'IMG']].copy()
    traditional_data = traditional_df[['Character', 'AoA', 'Familiarity', 'Concreteness', 'Imageability']].copy()

    # 初始化繁简转换器
    cc = OpenCC('t2s')  # 繁体转简体

    # 将繁体字符转换为简体
    traditional_data['S_Character'] = traditional_data['Character'].apply(
        lambda x: cc.convert(str(x.rstrip("1"))) if pd.notna(x.rstrip("1")) else x.rstrip("1")
    )

    # 重命名列以区分简体和繁体数据
    simplified_data.columns = ['S_Character', 'S_AoA', 'S_Familiarity', 'S_Concreteness', 'S_Imageability']
    traditional_data.columns = ['T_Character', 'T_AoA', 'T_Familiarity', 'T_Concreteness', 'T_Imageability', 'S_Character_temp']

    # 使用转换后的简体字符进行合并
    merged_data = pd.merge(
        simplified_data,
        traditional_data[
            ['S_Character_temp', 'T_Character', 'T_AoA', 'T_Familarity', 'T_Concreteness', 'T_Imageability']],
        left_on='S_Character',
        right_on='S_Character_temp',
        how='inner'
    )

    # 删除临时列
    # merged_data = merged_data.drop('S_Character_temp', axis=1)

    # 确保列的顺序正确
    final_columns = [
        'S_Character', 'S_AoA', 'S_Familiarity', 'S_Concreteness', 'S_Imageability',
        'T_Character', 'S_Character_temp', 'T_AoA', 'T_Familiarity', 'T_Concreteness', 'T_Imageability'
    ]
    merged_data = merged_data[final_columns]

    # 显示结果信息
    print(f"处理完成！共匹配到 {len(merged_data)} 条数据")
    print("\n前5行数据预览：")
    print(merged_data.head())

    # 保存结果到Excel文件
    merged_data.to_excel(r'merged.xlsx', index=False)
    print("\n结果已保存到 'merged.xlsx'")

    return merged_data

# 读取两个Excel文件, 输出不匹配数据，将繁体数据中简繁形式相同的字符数据作为few-shot prompt的case
def process_unmatched_data():
    # 读取简体中文数据
    simplified_df = pd.read_excel(r'Simplified_norm_2007.xlsx')
    traditional_df = pd.read_excel(r'Traditional_norm_2021.xlsx')

    # 提取需要的列
    simplified_data = simplified_df[['Word', 'AoA', 'FAM', 'CON', 'IMG']].copy()
    traditional_data = traditional_df[['Character', 'AoA', 'Familarity', 'Concreteness', 'Imageability']].copy()

    # 初始化繁简转换器
    cc = OpenCC('t2s')  # 繁体转简体

    # 将繁体字符转换为简体
    traditional_data['S_Character'] = traditional_data['Character'].apply(
        lambda x: cc.convert(str(x.rstrip("1"))) if pd.notna(x.rstrip("1")) else x.rstrip("1")
    )

    # 重命名列以区分简体和繁体数据
    simplified_data.columns = ['S_Character', 'S_AoA', 'S_Familarity', 'S_Concreteness', 'S_Imageability']
    traditional_data.columns = ['T_Character', 'T_AoA', 'T_Familarity', 'T_Concreteness', 'T_Imageability', 'S_Character_temp']

    # 使用左连接合并数据，这样可以保留所有简体数据
    merged_data_left = pd.merge(
        simplified_data,
        traditional_data[['S_Character_temp', 'T_Character', 'T_AoA', 'T_Familarity', 'T_Concreteness', 'T_Imageability']],
        left_on='S_Character',
        right_on='S_Character_temp',
        how='left'
    )

    # 使用右连接合并数据，这样可以保留所有繁体数据
    merged_data_right = pd.merge(
        simplified_data,
        traditional_data[['S_Character_temp', 'T_Character', 'T_AoA', 'T_Familarity', 'T_Concreteness', 'T_Imageability']],
        left_on='S_Character',
        right_on='S_Character_temp',
        how='right'
    )

    # 提取不匹配的简体数据（在繁体数据中找不到对应的）
    unmatched_simplified = merged_data_left[merged_data_left['S_Character_temp'].isna()].copy()
    unmatched_simplified = unmatched_simplified[['S_Character', 'S_AoA', 'S_Familarity', 'S_Concreteness', 'S_Imageability']]

    # 提取不匹配的繁体数据（在简体数据中找不到对应的）
    unmatched_traditional = merged_data_right[merged_data_right['S_Character'].isna()].copy()
    unmatched_traditional = unmatched_traditional[['S_Character_temp', 'T_Character', 'T_AoA', 'T_Familarity', 'T_Concreteness', 'T_Imageability']]
    unmatched_traditional.columns = ['S_Character', 'T_Character', 'T_AoA', 'T_Familarity', 'T_Concreteness', 'T_Imageability']

    # 输出不匹配的数据到文件
    unmatched_simplified.to_excel(r'Unmatched_Simplified.xlsx', index=False)
    unmatched_traditional.to_excel(r'Unmatched_Traditional.xlsx', index=False)

    print(f"不匹配的简体数据数量: {len(unmatched_simplified)}")
    print(f"不匹配的繁体数据数量: {len(unmatched_traditional)}")

    return unmatched_simplified, unmatched_traditional

# 输出匹配数据中简繁为1-m关系的case
def check_duplicates():
    try:
        # 读取结果文件
        result_df = pd.read_excel(r"@merged.xlsx")

        # 检查S_Character列的重复项
        duplicate_mask = result_df.duplicated('S_Character', keep=False)
        duplicate_data = result_df[duplicate_mask]

        if len(duplicate_data) > 0:
            print(
                f"发现 {len(duplicate_data)} 行重复数据（涉及 {duplicate_data['S_Character'].nunique()} 个不同的重复字符）")
            print("\n重复的S_Character及其出现次数：")

            # 统计每个重复字符的出现次数
            char_counts = result_df['S_Character'].value_counts()
            duplicate_chars = char_counts[char_counts > 1]

            for char, count in duplicate_chars.items():
                print(f"字符 '{char}': 出现 {count} 次")

            print("\n详细的重复数据：")
            print(duplicate_data.sort_values('S_Character'))

            # 保存重复数据到单独文件
            duplicate_data.to_excel(r"duplicates.xlsx", index=False)
            print(f"\n重复数据详情已保存到 'duplicates.xlsx'")

            return duplicate_data, result_df
        else:
            print("没有发现重复的S_Character")
            return None, result_df

    except FileNotFoundError:
        print("结果文件 'merged.xlsx' 未找到，请先运行处理程序")
        return None, None

def rewrite_jsonl_files(input_folder):# qwen, deepseek, gpt
    # 查找所有jsonl文件
    jsonl_files = glob.glob(os.path.join(input_folder, "**", "*.jsonl"), recursive=True)
    if not jsonl_files:
        jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))

    for file_path in jsonl_files:
        # 构建新文件名（添加_re后缀）
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name_without_ext, ext = os.path.splitext(base_name)
        new_file_path = os.path.join(dir_name, f"{name_without_ext}_re{ext}")

        print(f"处理: {file_path} -> {new_file_path}")

        # 读取所有行并处理
        with open(file_path, 'r', encoding='gbk') as f:
            lines = [line.lstrip('"').rstrip().rstrip('"').replace('\\"', '"') + '\n' for line in f if line.strip()]

        # 写回原文件
        with open(new_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    print(f"完成! 处理了 {len(jsonl_files)} 个文件")

def convert_jsonl_to_json(input_folder):
    # 查找所有jsonl文件
    jsonl_files = glob.glob(os.path.join(input_folder, "**", "*.jsonl"), recursive=True)
    if not jsonl_files:
        jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))

    for jsonl_file in jsonl_files:
        # 构建输出JSON文件路径
        json_file = os.path.splitext(jsonl_file)[0] + ".json"
        print(f"转换: {jsonl_file} -> {json_file}")

        # 读取JSONL并转换为JSON数组
        data_list = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # 跳过解析错误的行

        # 写入JSON文件
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"完成! 转换了 {len(jsonl_files)} 个文件")

def dedup_keep_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def rewrite_magistral_jsonl_files(input_folder):
    pattern = r'\{\\"Character\\":\s*\\"([^"]+)\\"\s*,\s*\\"AoA\\":\s*(\d+)\s*\}'
    extracted_data = []

    jsonl_files = glob.glob(os.path.join(input_folder, "**", "*.jsonl"), recursive=True)
    if not jsonl_files:
        jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))

    for file_path in jsonl_files:
        # 构建新文件名（添加_re后缀）
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name_without_ext, ext = os.path.splitext(base_name)
        new_file_path = os.path.join(dir_name, f"{name_without_ext}_re.json")

        print(f"处理: {file_path} -> {new_file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                matches = re.findall(pattern, content)
                print(matches)

                matches = dedup_keep_order(matches)

                for character, aoa in matches:
                    # 处理Unicode转义序列
                    try:
                        character = character.encode().decode('unicode_escape')
                    except:
                        pass

                    extracted_data.append({
                        "Character": character,
                        "AoA": int(aoa)
                    })

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

        # 保存结果
        with open(new_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(extracted_data, json_file, ensure_ascii=False, indent=2)

        print(f"提取完成! 共找到 {len(extracted_data)} 条记录")
        print(f"结果已保存到: {new_file_path}")

# 详细分析列表元素，返回：非空元素数量、元素类型统计、所有元素是否为字典的布尔值
def analyze_list_elements(data_list):
    if not isinstance(data_list, list):
        return None

    results = {
        'total_count': len(data_list),
        'non_empty_count': 0,
        'type_counter': Counter(),
        'all_are_dicts': True,
        'non_dict_items': [],
        'empty_items_count': 0
    }

    for idx, item in enumerate(data_list):
        # 统计非空元素
        if item:
            results['non_empty_count'] += 1
        else:
            results['empty_items_count'] += 1

        # 统计元素类型
        item_type = type(item).__name__
        results['type_counter'][item_type] += 1

        # 检查是否所有元素都是字典
        if not isinstance(item, dict):
            results['all_are_dicts'] = False
            results['non_dict_items'].append({
                'index': idx,
                'type': item_type,
                'value': str(item)[:100]  # 只取前100个字符
            })

    return results

# 完整的JSON文件分析函数（文件名解析、JSON有效性检查、元素数量统计、元素类型检查）
def analyze_json_files_complete(folder_path):
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    print(f"找到 {len(json_files)} 个JSON文件")
    print("=" * 80)

    # 汇总统计
    summary = {
        'total_files': len(json_files),
        'valid_json': 0,
        'invalid_json': 0,
        'is_list': 0,
        'all_dicts': 0,
        'traditional_2339': 0,
        'simple_2214': 0
    }

    for file_path in json_files:
        # 获取文件名（不含路径和扩展名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # 用"_"拆分文件名
        parts = file_name.split("_")

        # 解析文件名各部分
        if len(parts) >= 5:
            language, task, file_type, mode, model = parts[:5]
            extra_info = '_'.join(parts[5:]) if len(parts) > 5 else None
        else:
            # 如果不足5部分，用占位符
            language = parts[0] if len(parts) > 0 else "unknown"
            task = parts[1] if len(parts) > 1 else "unknown"
            file_type = parts[2] if len(parts) > 2 else "unknown"
            mode = parts[3] if len(parts) > 3 else "unknown"
            model = parts[4] if len(parts) > 4 else "unknown"
            extra_info = None

        print(f"文件: {file_name}")
        print(f"语言: {language}, 任务: {task}, 类型: {file_type}, 模式: {mode}, 模型: {model}")
        if extra_info:
            print(f"额外信息: {extra_info}")

        try:
            # 尝试读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            summary['valid_json'] += 1

            # 检查是否是列表
            if isinstance(data, list):
                summary['is_list'] += 1

                # 分析列表元素
                analysis = analyze_list_elements(data)

                if analysis:
                    # 输出分析结果
                    print(f"元素统计:")
                    print(f"总元素数量: {analysis['total_count']}")
                    print(f"非空元素数量: {analysis['non_empty_count']}")
                    print(f"空元素数量: {analysis['empty_items_count']}")

                    # 输出类型统计
                    print(f"元素类型统计:")
                    for type_name, count in analysis['type_counter'].most_common():
                        percentage = (count / analysis['total_count']) * 100
                        print(f"{type_name}: {count} ({percentage:.1f}%)")

                    # 检查是否所有元素都是字典
                    if analysis['all_are_dicts']:
                        print(f"所有元素都是字典类型")
                        summary['all_dicts'] += 1
                    else:
                        print(f"并非所有元素都是字典类型")
                        print(f"非字典元素数量: {len(analysis['non_dict_items'])}")
                        # 显示前5个非字典元素的信息
                        for item_info in analysis['non_dict_items'][:5]:
                            print(
                                f"索引 {item_info['index']}: 类型={item_info['type']}, 值={item_info['value']}")
                        if len(analysis['non_dict_items']) > 5:
                            print(f"还有 {len(analysis['non_dict_items']) - 5} 个非字典元素")

                    # 根据language进行特定检查
                    print(f"语言特定检查:")
                    if language == "traditional":
                        is_correct = analysis['total_count'] == 2339
                        result = f"通过 (2339个元素)" if is_correct else f"失败: 期望2339个元素，实际有{analysis['total_count']}个"
                        print(f"{result}")
                        if is_correct:
                            summary['traditional_2339'] += 1

                    elif language == "simple":
                        is_correct = analysis['total_count'] == 2214
                        result = f"通过 (2214个元素)" if is_correct else f"失败: 期望2214个元素，实际有{analysis['total_count']}个"
                        print(f"{result}")
                        if is_correct:
                            summary['simple_2214'] += 1

                    else:
                        print(f"元素数量: {analysis['total_count']}")

                else:
                    print(f"无法分析列表")

            else:
                print(f"JSON格式有效，但不是列表类型")
                print(f"数据类型: {type(data).__name__}")

        except json.JSONDecodeError as e:
            summary['invalid_json'] += 1
            print(f"无法解析为有效的JSON格式")
            print(f"错误信息: {str(e)[:100]}")

        except Exception as e:
            summary['invalid_json'] += 1
            print(f"读取文件时发生错误")
            print(f"错误信息: {str(e)}")

        print("-" * 80)

    # 输出汇总统计
    print("\n" + "=" * 80)
    print("汇总统计:")
    print(f"总文件数: {summary['total_files']}")
    print(f"有效JSON文件: {summary['valid_json']} ({summary['valid_json'] / summary['total_files'] * 100:.1f}%)")
    print(f"无效JSON文件: {summary['invalid_json']} ({summary['invalid_json'] / summary['total_files'] * 100:.1f}%)")
    print(f"列表类型文件: {summary['is_list']} ({summary['is_list'] / summary['valid_json'] * 100:.1f}% of valid)")
    print(
        f"所有元素都是字典的文件: {summary['all_dicts']} ({summary['all_dicts'] / summary['is_list'] * 100:.1f}% of lists)")

    if summary['is_list'] > 0:
        print(f"traditional文件通过2339检查: {summary['traditional_2339']}")
        print(f"simple文件通过2214检查: {summary['simple_2214']}")
    print("=" * 80)

# 将问题文件导出到CSV
def export_issues_to_csv(folder_path, output_file="json_issues.csv"):

    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    issues = []

    for file_path in json_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        parts = file_name.split("_")
        language = parts[0] if len(parts) > 0 else "unknown"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                analysis = analyze_list_elements(data)

                # 检查问题
                issues_found = []
                if not analysis['all_are_dicts']:
                    issues_found.append("包含非字典元素")

                if language in ["traditional"] and analysis['total_count'] != 2339:
                    issues_found.append(f"元素数量不为2339(实际{analysis['total_count']})")

                if language in ["simple"] and analysis['total_count'] != 2214:
                    issues_found.append(f"元素数量不为2214(实际{analysis['total_count']})")

                if analysis['empty_items_count'] > 0:
                    issues_found.append(f"包含{analysis['empty_items_count']}个空元素")

                if issues_found:
                    issues.append({
                        'file_name': file_name,
                        'language': language,
                        'total_elements': analysis['total_count'],
                        'non_empty_elements': analysis['non_empty_count'],
                        'all_dicts': analysis['all_are_dicts'],
                        'issues': '; '.join(issues_found)
                    })

        except json.JSONDecodeError:
            issues.append({
                'file_name': file_name,
                'language': language,
                'total_elements': 'N/A',
                'non_empty_elements': 'N/A',
                'all_dicts': 'N/A',
                'issues': '无效的JSON格式'
            })

    # 写入CSV
    if issues:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['file_name', 'language', 'total_elements', 'non_empty_elements', 'all_dicts', 'issues']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(issues)

        print(f"已导出 {len(issues)} 个问题文件到 {output_file}")

def check_sameTS():
    # 简化版本
    file_path = r'C:\Users\Floret\Desktop\Cantonese Norms\data\@2\@test.xlsx'

    # 读取文件
    df = pd.read_excel(file_path)

    # 检查并比较两列
    if 'T_Character' in df.columns and 'S_Character_temp' in df.columns:
        print(df['T_Character'])
        print(df['S_Character_temp'])
        # 计算对应位置相同的元素个数
        match_count = (df['T_Character'] == df['S_Character_temp']).sum()
        print(f"两列对应位置相同的元素个数: {match_count}")

        # 如果需要显示匹配的具体内容
        matching_rows = df[df['T_Character'] == df['S_Character_temp']]
        print(f"\n匹配的行 (共{match_count}行):")
        print(matching_rows[['T_Character', 'S_Character_temp']])
    else:
        print("错误: 文件中缺少所需的列")

# 预定义的正则表达式清洗规则
regex_patterns = {
        'common_qwen_error': [
            # 适用于实时生成
            # (r'.*\n*\{"', '{"'),
            # (r'\}\n*.*', '}'),
            # #处理前半部分
            # (r'"[^\n^\s]*\\n\{', '{'),
            # (r'"\{(\\)*"', '{"'),
            # (r'\\"', '"'),
            # # (r'\\"', '"'),
            # (r'\\n\}', '}'),
            # #处理后半部分
            # (r'\}\s*(\\n)+[^\n^\s]*', '},'),
            # # (r'\}\s*(\\n)+.*\n', '},'),
            # # (r',,[^\n]*,\n', ',\n'),
            # (r',[^\}^\n]*,', ','),
            # (r'\\n\s*"', '"'),
            # (r'\}",', '},'),
            # (r'\s*,', ','),
            # (r'\{,\s+"', '{"'),
            # (r',,\s+"', ', "'),
            # (r'\}', '},'),
            # (r'\},', '')
            # (r'\},\s*\n*\]', '}]')
        ]
    }

# 应用一组正则表达式清洗规则
def apply_regex_cleaners(text, patterns_list):
    cleaned_text = text
    for pattern, replacement in patterns_list:
        try:
            if callable(replacement):
                # 如果替换是函数
                cleaned_text = re.sub(pattern, replacement, cleaned_text)
            else:
                # 普通字符串替换
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.UNICODE)
        except Exception as e:
            print(f"正则表达式应用错误: {pattern} -> {replacement}, 错误: {e}")
    return cleaned_text

# 清洗单行JSON文本
def clean_json_line(line):
    original_line = line

    # 应用所有清洗规则
    for rule_name, patterns in regex_patterns.items():
        line = apply_regex_cleaners(line, patterns)
        # stats['patterns_applied'][rule_name] = stats['patterns_applied'].get(rule_name, 0) + 1

    return line

# 初步检查所有的json结果文件并按照模型典型错误，分类清洗文件，重新输出
def clean_json_data(folder_path, output_dir): # -> checked
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    print(f"找到 {len(json_files)} 个JSON文件")
    print("=" * 80)

    for file_path in json_files:
        # 获取文件名（不含路径和扩展名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # 用"_"拆分文件名
        language, task, type, mode, model = file_name.split("_")

        output_path = f"{output_dir}\\{language}_{task}_{type}_{mode}_{model}.json"

        try:
            with open(file_path, 'r', encoding='utf-8') as infile,\
                    open(output_path, 'w', encoding='utf-8') as outfile:

                for line_num, line in enumerate(infile, 1):
                    # stats['total_lines'] += 1

                    # if not line.strip():
                    # outfile.write(line)
                    # continue

                    print(line)

                    try:
                        # 尝试解析原始行（验证是否为有效JSON）
                        json.loads(line)
                        # 如果已经是有效JSON，可以不处理或轻度处理
                        cleaned_line = clean_json_line(line)
                    except json.JSONDecodeError:
                        # 如果不是有效JSON，尝试修复
                        cleaned_line = clean_json_line(line)

                    outfile.write(cleaned_line)
                    print(cleaned_line)

            # print(f"清洗完成！")
            # print(f"统计信息: {stats}")
            print(f"输出文件: {output_path}")

        except Exception as e:
            print(f"处理文件时出错: {e}")

def qwen7bc_T2S(infolder, outfolder):
    input_path = r'@merged.xlsx'

    # 加载字符列表
    traditional_input_dict = data_generation(input_path, 'traditional')
    traditional_list = list(traditional_input_dict.keys())
    simple_input_dict = data_generation(input_path, 'simple')
    simple_list = list(simple_input_dict.keys())
    json_files = glob.glob(os.path.join(infolder, "*.json"))

    cc = OpenCC('t2s')

    for file_path in json_files:
        print(f"处理文件: {file_path}")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        language, task, type_, mode, model = file_name.split("_")

        characters = traditional_list if language == 'traditional' else simple_list

        # 准备输出路径
        output_path = os.path.join(outfolder, f"{language}_{task}_{type_}_{mode}_{model}.json")
        output_list = []

        # 加载原始结果
        with open(file_path, 'r', encoding='utf-8') as infile:
            results = json.load(infile)

        n = 0
        for i, r in enumerate(results):
            if n >= len(characters):
                break

            character = characters[n].strip('1')
            # 只有当 n+1 在索引范围内时才获取 character1
            if n + 1 < len(characters):
                character1 = characters[n + 1].strip('1')
            else:
                # output_list.append({list(r.keys())[0]: character, list(r.keys())[1]: list(r.values())[1]})
                character1 = None

            if language == 'traditional':
                simple_form = list(r.values())[0].strip('1')
            else:
                simple_form = cc.convert(list(r.values())[0].strip('1'))

            # 检查当前结果是否有效
            is_valid = (isinstance(r, dict) and
                        simple_form != "NaN" and
                        list(r.values())[1] in [1, 2, 3, 4, 5, 6, 7])

            if is_valid:
                if simple_form == character:
                    output_list.append({list(r.keys())[0]: character, list(r.keys())[1]: list(r.values())[1]})
                    n += 1
                elif character1 is not None and simple_form == character1:
                    # 只有当 character1 存在时才执行这个分支
                    output_list.append({list(r.keys())[0]: character, list(r.keys())[1]: "NaN"})
                    output_list.append({list(r.keys())[0]: character1, list(r.keys())[1]: list(r.values())[1]})
                    n += 2
                else:
                    output_list.append({list(r.keys())[0]: character, list(r.keys())[1]: list(r.values())[1]})
                    print(f"{simple_form}, {character}")
                    n += 1
            else:
                output_list.append({list(r.keys())[0]: character, list(r.keys())[1]: "NaN"})
                n += 1

        # 保存结果
        with open(output_path, "w", encoding='utf-8') as outfile:
            print(len(output_list))
            json.dump(output_list, outfile, ensure_ascii=False, indent=2)

# 核对数量和顺序：在checked的基础上检查（并重新输出）json格式
def check_format_number(infolder, outfolder):
    input_path = r'@merged.xlsx'

    # 加载字符列表
    traditional_input_dict = data_generation(input_path, 'traditional')
    traditional_list = list(traditional_input_dict.keys())
    simple_input_dict = data_generation(input_path, 'simple')
    simple_list = list(simple_input_dict.keys())

    json_files = glob.glob(os.path.join(infolder, "*.json"))

    for file_path in json_files:
        print(f"处理文件: {file_path}")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        language, task, type_, mode, model = file_name.split("_")

        # 选择字符列表
        characters = traditional_list if language == 'traditional' else simple_list

        # 准备输出路径
        output_path = os.path.join(outfolder, f"{language}_{task}_{type_}_{mode}_{model}.json")

        # 加载原始结果
        with open(file_path, 'r', encoding='utf-8') as infile:
            results = json.load(infile)
        # if len(results) != 2339:
        # print(f"{file_name}-{language}-{len(results)}")


        final_results = []
        valid_count = 0
        original_valid_count = 0

        print(f"处理文件: {file_path}, 原始结果数: {len(results)}")

        # 模型函数映射，避免重复的if-elif
        model_functions = {
            'qwen': lambda c: qwen_bailian_2(type_, task, mode, c),
            'deepseek': lambda c: deepseek_bailian_2(type_, task, mode, c),
            'llama': lambda c: llama_openrouter_2(type_, task, mode, c),
            'gpt': lambda c: gpt_openrouter_2(type_, task, mode, c),
            'mistral': lambda c: mistral_openrouter_2(type_, task, mode, c),
            'qwen7b': lambda c: qwen_7b(type_, task, mode, c),
            'qwen7bc': lambda c: qwen_7b_cantonese(type_, task, mode, c)
        }

        # 逐个检查并处理
        for i, r in enumerate(results):
            if i >= len(characters):
                break

            character = characters[i].strip('1')

            # 检查当前结果是否有效
            is_valid = (isinstance(r, dict) and
                        list(r.values())[0].strip('1') == character and
                        list(r.values())[1] in [1, 2, 3, 4, 5, 6, 7])

            if is_valid:
                # print(f"{i}-{r}-{character}")
                # 原始结果有效，直接使用并转换键
                print('Success!')
                value = list(r.values())[1]
                final_results.append({"Character": character, task.capitalize(): value})
                valid_count += 1
                original_valid_count += 1
            else:
                print('Failure!')
                # 原始结果无效，重新生成
                # if model in model_functions:
                result = model_functions[model](character)
                # else:
                #     print(f"错误: 未知模型 {model}, 文件: {file_path}, 字符: {character}")
                #     result = {"Character": 'NaN', task.capitalize(): 'NaN'}

                # 验证并处理结果
                if isinstance(result, str):
                    #简单清洗格式 提高输出率
                    for rule_name, patterns in regex_patterns.items():
                        result = apply_regex_cleaners(result, patterns)

                    try:
                        result_dict = json.loads(result)
                        final_results.append(result_dict)
                        if result_dict.get("Character") != "NaN":
                            valid_count += 1
                        print(f"JSON解析正确: {character}, 内容: {result}")
                    except json.JSONDecodeError:
                        print(f"JSON解析错误: {character}, 内容: {result}")
                        final_results.append(result)
                elif isinstance(result, dict):
                    final_results.append(result)
                    if result.get("Character") != "NaN":
                        valid_count += 1
                else:
                    print(f"未知结果类型: {type(result)}, 字符: {character}")
                    final_results.append(result)

        # 输出统计信息
        print(f"处理完成: {file_path}")
        print(f"总结果数: {len(final_results)}")
        print(f"原结果中的有效条目: {original_valid_count}")
        print(f"现结果中的有效条目: {valid_count}")
        print(f"现结果中的无效条目: {len(characters) - valid_count}")

        # 保存结果
        with open(output_path, "w", encoding='utf-8') as outfile:
            json.dump(final_results, outfile, ensure_ascii=False, indent=2)


# 辅助函数：检查字典是否符合特定格式
def is_valid_result(infolder):
    input_path = r'@merged.xlsx'

    # 加载字符列表
    traditional_input_dict = data_generation(input_path, 'traditional')
    traditional_list = list(traditional_input_dict.keys())
    simple_input_dict = data_generation(input_path, 'simple')
    simple_list = list(simple_input_dict.keys())

    json_files = glob.glob(os.path.join(infolder, "*.json"))

    sum = 0
    for file_path in json_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        language, task, type_, mode, model = file_name.split("_")

        # 选择字符列表
        characters = traditional_list if language == 'traditional' else simple_list

        # 加载原始结果
        with open(file_path, 'r', encoding='utf-8') as infile:
            results = json.load(infile)
        # if len(results) != 2339:
        print(f"{file_name}-{language}-{len(results)}")

        n = 0
        # 逐个检查并处理
        for i, r in enumerate(results):
            if i >= len(characters):
                break

            character = characters[i].strip('1')

            # 检查当前结果是否有效
            is_valid = (isinstance(r, dict) and
                        list(r.values())[0].strip('1') == character and
                        list(r.values())[1] in [1, 2, 3, 4, 5, 6, 7])

            if not is_valid:
                print(f"{i}-{r}-{character}")
                n += 1

        print(f"现结果中的无效条目: {n}")
        sum += n
    print(f"现结果中的总无效条目: {sum}")

def restructure_round_dict(input_folder):
    '''
      {"traditional":[{                                 # 1. language
          "id": @                                       # 2.1 id
          "num": @                                      # 2.2 number (主要针对repeated simple character)
          "character": @                                # 2.3 character
          "results": {
            "AoA":{                                     # 3 task
              "zero":{                                  # 4 mode
                "en":{                                  # 5 type
                  "qwen": [r1, r2, r3, avg]             # 6 model [round1, round2, round3, average]
                  "deepseek": [r1, r2, r3, avg]
                  "llame": [r1, r2, r3, avg]
                  "gpt": [r1, r2, r3, avg]
                }
                "zh":{...}
                "ca":{...}
              }
              "few":{...}
            }
            "imageability":{...}
            "concreteness":{...}
            "familiarity":{...}
          }...]

      "simple":{
          "id": @
          "num": @
          "character": @
          "results": {
          ...
          }
        }...]
      }
    '''
    # 定义一个总的字典用于储存数据
    sum_dic = {'traditional': {}, 'simple': {}}
    character_info = {'traditional': {}, 'simple': {}}  # 存储字符的额外信息

    # 准备输出路径
    output_path = os.path.join(input_folder, f"round1.json")

    # 查找所有jsonl文件
    jsonl_files = glob.glob(os.path.join(input_folder, "**", "*.json"), recursive=True)
    if not jsonl_files:
        jsonl_files = glob.glob(os.path.join(input_folder, "*.json"))

    # 提取文件名信息
    for file_path in jsonl_files:
        base_name = os.path.basename(file_path)
        name_without_ext, ext = os.path.splitext(base_name)
        language1, task1, type1, mode1, model1 = name_without_ext.split('_')

        with open(file_path, 'r', encoding='utf-8') as fin:
            results = json.load(fin)

        for i, item in enumerate(results):
            character = item.get('Character', '')
            if not character:
                continue

            # 初始化字符条目
            if character not in sum_dic[language1]:
                sum_dic[language1][character] = {
                    'id': i + 1,
                    'num': 1,  # 如果有重复字符，需要调整
                    'character': character,
                    'results': {}
                }

            # 确保results中的嵌套结构存在
            current = sum_dic[language1][character]['results']

            # 初始化task1
            if task1 not in current:
                current[task1] = {}

            # 初始化mode1
            if mode1 not in current[task1]:
                current[task1][mode1] = {}

            # 初始化type1
            if type1 not in current[task1][mode1]:
                current[task1][mode1][type1] = {}

            # 初始化model1的列表
            if model1 not in current[task1][mode1][type1]:
                current[task1][mode1][type1][model1] = []

            # 添加结果值
            # 注意：item[task1.capital()] 可能不存在，需要检查
            task_value_key = task1.capitalize()
            if task_value_key in item:
                value = item[task_value_key]
                # 如果是NaN，可能需要处理为None或特定值
                if value == 'NaN':
                    value = None
                current[task1][mode1][type1][model1].append(value)

    # 转换为列表格式并排序
    for language in ['traditional', 'simple']:
        # 将字典转换为列表
        char_list = list(sum_dic[language].values())
        # 按id排序
        char_list.sort(key=lambda x: x['id'])
        sum_dic[language] = char_list

    # 保存结果
    with open(output_path, "w", encoding='utf-8') as outfile:
        json.dump(sum_dic, outfile, ensure_ascii=False, indent=2)

    return sum_dic

def restructure_round_dict_v2(input_folder):
    # 使用defaultdict简化初始化
    sum_dic = defaultdict(lambda: defaultdict(lambda: {
        'id': 0,
        'num': 1,
        'character': '',
        'results': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    }))

    # 准备输出路径
    output_path = os.path.join(input_folder, f"round1.json") #round2.json #round3.json

    # 处理文件
    for file_path in glob.glob(os.path.join(input_folder, "**", "*.json"), recursive=True):
        base_name = os.path.basename(file_path)
        name_without_ext, ext = os.path.splitext(base_name)
        language1, task1, type1, mode1, model1 = name_without_ext.split('_')

        with open(file_path, 'r', encoding='utf-8') as fin:
            results = json.load(fin)

        for i, item in enumerate(results):
            character = item.get('Character', '')
            if not character:
                continue

            # 设置基本信息
            char_data = sum_dic[language1][character]
            if char_data['id'] == 0:  # 第一次遇到这个字符
                char_data['id'] = i + 1
                char_data['character'] = character

            # 添加结果
            task_value = item.get(task1.capitalize())
            if task_value and task_value != 'NaN':
                char_data['results'][task1][mode1][type1][model1].append(task_value)

    # 转换和输出
    final_dict = {}
    for language in ['traditional', 'simple']:
        # 转换为列表并排序
        char_list = sorted(
            [data for data in sum_dic[language].values() if data['id'] > 0],
            key=lambda x: x['id']
        )
        final_dict[language] = char_list

    # 保存结果
    with open(output_path, "w", encoding='utf-8') as outfile:
        json.dump(sum_dic, outfile, ensure_ascii=False, indent=2)

    return final_dict

#合并多个round的结果，计算平均值。目标格式: "qwen": [r1, r2, r3, avg]
def merge_rounds(input_folder, round_names=['round1', 'round2', 'round3']):
    # 加载所有round的数据
    round_data = {}
    for round_name in round_names:
        round_file = os.path.join(input_folder, f"{round_name}.json")
        if os.path.exists(round_file):
            with open(round_file, 'r', encoding='utf-8') as f:
                round_data[round_name] = json.load(f)
        else:
            print(f"警告: 未找到文件 {round_file}")
            round_data[round_name] = {'simple': {}, 'traditional': {}}

    # 如果所有round都为空，直接返回
    if not any(round_data.values()):
        return None

    # 创建合并后的数据结构
    merged_data = {'simple': {}, 'traditional': {}}

    # 定义模型名称列表
    # models = ['qwen', 'deepseek', 'gpt', 'llama', 'mistral']
    models = ['qwen7b', 'qwen7bc']

    # 遍历每个语言
    for language in ['simple', 'traditional']:
        # 获取所有round中该语言的字符数据
        char_data_by_round = {}

        # 收集每个round的字符数据，按character索引
        for round_name in round_names:
            if round_name in round_data and language in round_data[round_name]:
                for character, char_data in round_data[round_name][language].items():
                    # character = char_data['character']
                    if character not in char_data_by_round:
                        char_data_by_round[character] = {}
                    char_data_by_round[character][round_name] = char_data

        # 合并字符数据
        for character, rounds_dict in char_data_by_round.items():
            # 获取第一个round的数据作为基础
            first_round = round_names[0]
            if first_round not in rounds_dict:
                continue

            base_data = rounds_dict[first_round]

            # 创建合并后的字符条目
            merged_char = {
                'id': base_data.get('id', 0),
                'num': base_data.get('num', 1),  # 如果有重复字符，需要调整
                'character': character,
                'results': {}
            }

            # 遍历所有任务
            for task in base_data['results'].keys():
                merged_char['results'][task] = {}

                # 遍历所有mode
                for mode in base_data['results'][task].keys():
                    merged_char['results'][task][mode] = {}

                    # 遍历所有type
                    for type_ in base_data['results'][task][mode].keys():
                        merged_char['results'][task][mode][type_] = {}

                        # 遍历所有模型
                        for model in models:
                            model_results = []

                            # 收集每个round的结果
                            for round_name in round_names:
                                if (round_name in rounds_dict and
                                        task in rounds_dict[round_name]['results'] and
                                        mode in rounds_dict[round_name]['results'][task] and
                                        type_ in rounds_dict[round_name]['results'][task][mode] and
                                        model in rounds_dict[round_name]['results'][task][mode][type_]):

                                    round_values = rounds_dict[round_name]['results'][task][mode][type_][model]
                                    if isinstance(round_values, list) and len(round_values) > 0:
                                        # 如果有多个值，取最后一个（假设是平均值）或第一个
                                        model_results.append(
                                            round_values[-1] if len(round_values) > 1 else round_values[0])
                                    else:
                                        model_results.append(None)
                                else:
                                    model_results.append(None)

                            # 计算平均值（排除None值）
                            valid_results = [r for r in model_results if r is not None and r != 'NaN']
                            if valid_results:
                                try:
                                    # 尝试转换为数字计算平均值
                                    numeric_results = []
                                    for r in valid_results:
                                        if isinstance(r, (int, float)):
                                            numeric_results.append(r)
                                        elif isinstance(r, str):
                                            try:
                                                numeric_results.append(float(r))
                                            except ValueError:
                                                pass

                                    if numeric_results:
                                        avg = sum(numeric_results) / len(numeric_results)
                                        # 保留两位小数
                                        avg = round(avg, 6)
                                    else:
                                        avg = 'NaN'
                                except Exception:
                                    avg = 'NaN'
                            else:
                                avg = 'NaN'

                            # 添加平均值到结果列表
                            model_results.append(avg)
                            merged_char['results'][task][mode][type_][model] = model_results

            merged_data[language][character] = merged_char

    # 按id排序
    for language in ['simple', 'traditional']:
        if language in merged_data:
            # 获取该语言的所有条目，并按id排序
            sorted_items = sorted(
                merged_data[language].items(),  # 获取(键, 值)对
                key=lambda x: x[1]['id']  # 按值的'id'字段排序
            )

            # 创建新的有序字典
            merged_data[language] = OrderedDict(sorted_items)

    # 保存合并后的结果
    output_file = os.path.join(input_folder, "merged_rounds.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    return merged_data

def update_duplicate_nums(merged_data_path, excel_path, output_path=None):
    """
    根据excel中的重复统计结果，更新merged_data中simple部分character的num字段
    根据excel中的重复统计结果，更新merged_data中tradition部分traditional_character的simple_character字段
    适用于merge_rounds函数生成的merged_data，其中每个字符只有一条记录

    参数:
    merged_data_path: merged_rounds.json的路径
    excel_path: 包含重复统计的excel文件路径
    output_path: 输出文件路径，默认为原路径加上"_updated"
    """

    # 1. 读取excel文件，统计重复字符的出现次数
    try:
        result_df = pd.read_excel(excel_path)

        # 检查S_Character列的重复项
        duplicate_mask = result_df.duplicated('S_Character', keep=False)
        duplicate_data = result_df[duplicate_mask]

        if len(duplicate_data) > 0:
            print(
                f"发现 {len(duplicate_data)} 行重复数据（涉及 {duplicate_data['S_Character'].nunique()} 个不同的重复字符）")
            print("\n重复的S_Character及其出现次数：")

            # 统计每个字符的出现次数（包括非重复的）
            char_counts = result_df['S_Character'].value_counts()

            # 显示重复字符
            duplicate_chars = char_counts[char_counts > 1]
            for char, count in duplicate_chars.items():
                print(f"字符 '{char}': 出现 {count} 次")

            # 创建所有字符到出现次数的映射字典
            char_count_dict = char_counts.to_dict()
        else:
            print("没有发现重复字符")
            char_count_dict = {}

        # 建立繁体到简体的映射字典
        # 假设excel中有T_Character和S_Character_temp列
        t_s_mapping = {}

        # 检查是否有S_Character_temp列
        if 'S_Character_temp' in result_df.columns:
            for _, row in result_df.iterrows():
                trad_char = row.get('T_Character', '').strip("1")
                simp_char = row.get('S_Character_temp', '').strip("1")

                if trad_char and pd.notna(trad_char) and simp_char and pd.notna(simp_char):
                    t_s_mapping[trad_char] = simp_char

            print(f"\n建立了 {len(t_s_mapping)} 个繁简字符映射")
        else:
            print("警告：excel文件中没有找到S_Character_temp列，无法建立繁简映射")

    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None

    # 2. 读取merged_data
    try:
        with open(merged_data_path, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
    except Exception as e:
        print(f"读取merged_data文件时出错: {e}")
        return None

    # 3. 更新simple部分中所有字符的num字段
    updated_count = 0
    if 'simple' in merged_data:
        for character, char_data in merged_data['simple'].items():
            # character = char_data.get('character', '')

            # 如果字符在统计字典中，更新num字段
            if character in char_count_dict:
                # 获取该字符的出现次数
                occurrence_count = char_count_dict[character]

                # 检查是否需要更新
                original_num = char_data.get('num', 1)
                if original_num != occurrence_count:
                    char_data['num'] = occurrence_count
                    updated_count += 1
                    if occurrence_count > 1:
                        print(f"更新字符 '{character}': num从 {original_num} 改为 {occurrence_count} (重复字符)")
                    else:
                        print(f"更新字符 '{character}': num从 {original_num} 改为 {occurrence_count}")
                elif occurrence_count > 1:
                    print(f"字符 '{character}' 已正确标记为重复: num={original_num}")

    print(f"\n总共更新了 {updated_count} 个字符的num字段")

    # 4. 更新tradition部分的simple_character字段
    trad_updated_count = 0
    if 'traditional' in merged_data and t_s_mapping:
        for trad_character, char_data in merged_data['traditional'].items():
            # trad_character = char_data.get('traditional_character', '')

            # 如果繁体字在映射字典中，添加或更新simple_character字段
            if trad_character in t_s_mapping:
                simp_character = t_s_mapping[trad_character]

                # 检查是否需要更新
                original_simp = char_data.get('simple_character', '')
                if original_simp != simp_character:
                    char_data['simple_character'] = simp_character
                    trad_updated_count += 1
                    print(
                        f"更新繁体字符 '{trad_character}': simple_character从 '{original_simp}' 改为 '{simp_character}'")
                else:
                    print(f"繁体字符 '{trad_character}' 的simple_character已正确设置为 '{simp_character}'")
            elif trad_character:
                print(f"警告：繁体字符 '{trad_character}' 在excel中没有找到对应的简体字符映射")

    print(f"\n总共更新了 {trad_updated_count} 个繁体字符的simple_character字段")

    # 5. 统计重复字符数量
    if 'simple' in merged_data:
        duplicate_chars_in_data = [char_data for char_data in list(merged_data['simple'].values())
                                   if char_data.get('num', 1) > 1]
        print(f"merged_data中有 {len(duplicate_chars_in_data)} 个重复字符")

    # 5. 统计重复字符数量
    if 'simple' in merged_data:
        duplicate_chars_in_data = [char_data for char_data in list(merged_data['simple'].values())
                                   if char_data.get('num', 1) > 1]
        print(f"merged_data中有 {len(duplicate_chars_in_data)} 个重复字符")

    # 6. 保存更新后的数据
    if output_path is None:
        # 默认在原文件名后添加"_updated"
        dir_name = os.path.dirname(merged_data_path)
        base_name = os.path.basename(merged_data_path)
        name_without_ext, ext = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name_without_ext}_updated{ext}")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"更新后的数据已保存到: {output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return None

    return merged_data

#将整体结果数据拆分为简繁同体（1404）和简繁异体（935）
def split_TS(inpath, merged_data_path, outpath):
    # 加载字符列表
    simple_input_dict = data_generation(inpath, 'simple')
    simple_list = list(simple_input_dict.keys())
    traditional_input_dict = data_generation(inpath, 'traditional')
    traditional_list = list(traditional_input_dict.keys())

    with open(merged_data_path, 'r', encoding='utf-8') as fin:
        merged_data = json.load(fin)

    new_dic = {}
    new_simple_dic = {}
    new_traditional_dic = {}

    simple_dic = merged_data["simple"]
    traditional_dic = merged_data["traditional"]

    for s in simple_list:
        new_simple_dic[s] = simple_dic[s.strip("1")]
    print(len(list(new_simple_dic.keys())))

    for t in traditional_list:
        new_traditional_dic[t] = traditional_dic[t.strip("1")]
    print(len(list(new_traditional_dic.keys())))

    new_dic["simple"] = new_simple_dic
    new_dic["traditional"] = new_traditional_dic

    with open(outpath, 'w', encoding='utf-8') as fout:
        json.dump(new_dic, fout, ensure_ascii=False, indent=2)
    print(f"更新后的数据已保存到: {outpath}")

    return new_dic

def combine_freq_stroke_DeepSeek_GPT(outpath):
    # for overall data
    # ori_merge_path = r"@merged.xlsx"
    # raw_traditional_path = r"Traditional_norm_2021.xlsx"
    # raw_simple_path = r"Simplified_norm_2007.xlsx"
    # e1_results_path = r"merged_rounds_updated.json"

    # for same data
    ori_merge_path = r"@TS_same.xlsx"
    raw_traditional_path = r"Traditional_norm_2021.xlsx"
    raw_simple_path = r"Simplified_norm_2007.xlsx"
    e1_results_path = r"experiment1_same.json"

    # # for different data
    # ori_merge_path = r"@TS_different.xlsx"
    # raw_traditional_path = r"Traditional_norm_2021.xlsx"
    # raw_simple_path = r"Simplified_norm_2007.xlsx"
    # e1_results_path = r"experiment1_different.json"

    # 加载字符列表
    traditional_input_dict = data_generation(ori_merge_path, 'traditional')
    traditional_list = list(traditional_input_dict.keys())
    # print(len(traditional_list))
    simple_input_dict = data_generation(ori_merge_path, 'simple')
    simple_list = list(simple_input_dict.keys())
    # print(len(simple_list))

    # 读取数据
    raw_traditional_pd = pd.read_excel(raw_traditional_path)
    raw_simple_pd = pd.read_excel(raw_simple_path)
    ori_merge_pd = pd.read_excel(ori_merge_path)

    # 新增四列
    ori_merge_pd["T_Freq.log"] = np.nan
    ori_merge_pd["T_Stroke"] = np.nan
    ori_merge_pd["S_Freq.log"] = np.nan
    ori_merge_pd["S_Stroke"] = np.nan

    def get_freq_stroke(ori_merge_pd):
        # 处理繁体字频率和笔画信息
        for idx, row in ori_merge_pd.iterrows():
            t_char = row.get('T_Character', '')
            if pd.notna(t_char) and t_char in traditional_list:
                # 在raw_traditional_pd中查找
                trad_match = raw_traditional_pd[raw_traditional_pd['Character'] == t_char]
                if not trad_match.empty:
                    ori_merge_pd.at[idx, "T_Freq.log"] = trad_match.iloc[0].get('Frequency.Log', np.nan)
                    ori_merge_pd.at[idx, "T_Stroke"] = trad_match.iloc[0].get('#Stroke', np.nan)

            # 处理简体字频率和笔画信息
            s_char = row.get('S_Character', '')
            if pd.notna(s_char) and s_char in simple_list:
                # 在raw_simple_pd中查找
                simple_match = raw_simple_pd[raw_simple_pd['Word'] == s_char]
                if not simple_match.empty:
                    ori_merge_pd.at[idx, "S_Freq.log"] = simple_match.iloc[0].get('Log.CF', np.nan)
                    ori_merge_pd.at[idx, "S_Stroke"] = simple_match.iloc[0].get('NS', np.nan)

        return ori_merge_pd

    def load_model_data(inpath=e1_results_path, language='traditional', task='AoA', mode='zero', _type='en', model='deepseek'):
        score_list = []
        parser = MergedDataParser(inpath)

        if language == 'traditional':
            for c in traditional_list:
                print(c)
                number = parser.get_num_value(language, c)
                print(number)
                scores = parser.get_values(language, c.strip('1'), task, mode, _type, model)
                print(scores)
                avg_score = scores[3]  # 根据描述，第4个值是avg_score
                for _ in range(number):
                    score_list.append(avg_score)
        elif language == 'simple':
            for c in simple_list:
                number = parser.get_num_value(language, c)
                scores = parser.get_values(language, c.strip('1'), task, mode, _type, model)
                avg_score = scores[3]
                for _ in range(number):
                    score_list.append(avg_score)
        else:
            print("Wrong language!")

        print(f"Loaded {len(score_list)} scores for {model}_{language}_{task}_{mode}_{_type}")
        return np.array(score_list)

    def results_compute_AE(ori_merge_pd):
        # 创建新的DataFrame来收集所有列，避免碎片化
        new_columns = {}

        # 首先复制原始数据
        for col in ori_merge_pd.columns:
            new_columns[col] = ori_merge_pd[col].values

        for model in ["qwen", "deepseek", "gpt", "mistral", "llama"]:
        # for model in ['qwen7b', 'qwen7bc']:
            for language in ["traditional", "simple"]:
                for task in ["AoA", "familiarity", "concreteness", "imageability"]:
                    for mode in ["zero", "few"]:
                        for _type in ["en", "zh", "ca"]:
                            # 获取黄金标准分数列
                            if task == 'AoA':
                                gold_column = f"{language[0].capitalize()}_{task}"
                            else:
                                gold_column = f"{language[0].capitalize()}_{task.capitalize()}"
                            if gold_column in ori_merge_pd.columns:
                                gold_scores = ori_merge_pd[gold_column].values
                            else:
                                print(f"Warning: Column {gold_column} not found. Skipping...")
                                continue

                            # 获取模型预测分数
                            scores = load_model_data(e1_results_path, language, task, mode, _type, model)

                            # 确保长度一致
                            if len(scores) != len(gold_scores):
                                min_len = min(len(scores), len(gold_scores))
                                scores = scores[:min_len]
                                gold_scores = gold_scores[:min_len]

                            # 添加预测分数列
                            score_column = f"{model}_{language}_{task}_{mode}_{_type}"
                            new_columns[score_column] = np.full(len(ori_merge_pd), np.nan)
                            new_columns[score_column][:len(scores)] = scores[:len(ori_merge_pd)]

                            # 计算绝对误差并添加列
                            absolute_errors = np.abs(gold_scores - scores)
                            ae_column = f"{model}_{language}_{task}_{mode}_{_type}_ae"
                            new_columns[ae_column] = np.full(len(ori_merge_pd), np.nan)
                            new_columns[ae_column][:len(absolute_errors)] = absolute_errors[:len(ori_merge_pd)]

        # 一次性创建新的DataFrame
        new_pd = pd.DataFrame(new_columns)
        return new_pd

    # 执行函数
    add_freq_stroke = get_freq_stroke(ori_merge_pd)
    add_results_ae = results_compute_AE(add_freq_stroke)

    # 保存结果
    # 根据outpath的扩展名决定保存格式
    if outpath.endswith('.xlsx'):
        add_results_ae.to_excel(outpath, index=False)
    elif outpath.endswith('.csv'):
        add_results_ae.to_csv(outpath, index=False)
    else:
        # 默认保存为Excel
        add_results_ae.to_excel(outpath, index=False)

    print(f"Results saved to: {outpath}")
    return add_results_ae

# 抽取文件中absolute error排名前5和后5的字符
def read_combined_extract_character(target, inpath):
    # 读取Excel文件
    raw_combined_pd = pd.read_excel(inpath)

    results = {}  # 用于存储结果

    # target = "qwen7b_traditional_AoA"

    # 找出所有需要处理的列
    target_columns = []
    for col in raw_combined_pd.columns:
        # 检查列名是否满足两个条件之一
        if "_ae" in col and target in col:
            target_columns.append(col)

    if not target_columns:
        print("没有找到符合条件的列")
        return results

    # 创建新的列用于存储所有目标列的和
    sum_col_name = "combined_sum"
    raw_combined_pd[sum_col_name] = 0

    # 对每一行，将所有目标列的值相加
    for col in target_columns:
        raw_combined_pd[sum_col_name] += raw_combined_pd[col].fillna(0)

    raw_combined_pd[sum_col_name] = raw_combined_pd[sum_col_name] / len(target_columns)

    # 创建一个副本用于排序，同时保留原始索引对应关系
    df_sorted = raw_combined_pd.sort_values(by=sum_col_name, ascending=True).reset_index(drop=True)
    # print(df_sorted)

    # 获取排序后的前6个值和最后6个值的索引
    first_six_indices = df_sorted.head(6).index
    last_six_indices = df_sorted.tail(6).index

    # 提取对应的S_Character和T_Character
    first_six = []
    last_six = []

    if 'simple' in target:
        for idx in first_six_indices:
            character = df_sorted.loc[idx, 'S_Character']
            value = df_sorted.loc[idx, sum_col_name]
            first_six.append((character, value))

        for idx in last_six_indices:
            character = df_sorted.loc[idx, 'S_Character']
            value = df_sorted.loc[idx, sum_col_name]
            last_six.append((character, value))
    elif 'traditional' in target:
        for idx in first_six_indices:
            character = df_sorted.loc[idx, 'T_Character']
            value = df_sorted.loc[idx, sum_col_name]
            first_six.append((character, value))

        for idx in last_six_indices:
            character = df_sorted.loc[idx, 'T_Character']
            value = df_sorted.loc[idx, sum_col_name]
            last_six.append((character, value))
        else:
            print("wrong language")

    # 将结果存储到字典中
    results = {
        'target':target,
        'target_columns': target_columns,
        'sum_column': sum_col_name,
        'first_six': first_six,
        'last_six': last_six,
        'sorted_dataframe': df_sorted  # 可选：包含排序后的完整数据
    }

    print(f"处理了 {len(target_columns)} 个列:")
    for col in target_columns:
        print(f"  - {col}")
    print(last_six)
    # print(f"总和列: {sum_col_name}")

    # print(results)
    return results


if __name__ == '__main__':
    pass





