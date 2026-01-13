# -*- coding:utf-8 -*-
'''
@file   :visulization.py
@author :Floret Huacheng SONG
@time   :9/12/2025 下午7:43
@purpose: for visualizing the results data
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns
import pandas as pd
import re
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

#读取MAE Excel文件并构建数据字典
def read_MAEexcel_to_dict(file_path, language, metric, model_number):

    # 读取Excel文件
    df = pd.read_excel(file_path, header=None)

    if model_number == 5:
        # 定义模型名称顺序
        models = ['qwen', 'deepseek', 'gpt', 'mistral', 'llama']
    elif model_number == 2:
        models = ['qwen7b', 'qwen7b_ca']
    else:
        pass
    languages = ['en', 'zh', 'ca']

    # 初始化结果字典
    result = {}

    # 根据语言和指标查找数据
    language_found = False
    metric_found = False
    start_row = 0

    # 遍历查找语言和指标位置
    for i in range(len(df)):
        cell_value = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""

        # 查找语言
        if language in cell_value and not language_found:
            language_found = True
            continue

        # 查找指标
        if language_found and metric in cell_value and not metric_found:
            metric_found = True
            start_row = i + 1
            break

    if not metric_found:
        raise ValueError(f"未找到 {language} 的 {metric} 数据")

    # 读取数据
    if language == 'Cantonese':
        # 粤语有5行数据，每行对应一个模型的6个数据
        data_rows = 5
        data = df.iloc[start_row:start_row + data_rows, :6].values

        for i, model in enumerate(models):
            if i < len(data):
                model_data = {}

                # 粤语部分只需要一个duplicate层
                duplicate_data = {}

                # 前3个数据是zero_shot
                zero_shot_data = {}
                for k, lang in enumerate(languages):
                    zero_shot_data[lang] = float(data[i, k])
                duplicate_data['zero_shot'] = zero_shot_data

                # 后3个数据是few_shot
                few_shot_data = {}
                for k, lang in enumerate(languages):
                    few_shot_data[lang] = float(data[i, k + 3])
                duplicate_data['few_shot'] = few_shot_data

                model_data['duplicate'] = duplicate_data
                model_data['deduplicate'] = {}

                result[model] = model_data

    elif language == 'Mandarin':
        # 普通话有10行数据，每两行对应一个模型
        data_rows = 10
        data = df.iloc[start_row:start_row + data_rows, :6].values

        # 每两行对应一个模型
        for i, model in enumerate(models):
            row1_idx = i * 2
            row2_idx = i * 2 + 1

            if row1_idx < data_rows and row2_idx < data_rows:
                model_data = {}

                # 第一行作为duplicate
                duplicate_data = {}
                duplicate_zero_shot = {}
                duplicate_few_shot = {}

                # duplicate行的前3个数据是zero_shot
                for k, lang in enumerate(languages):
                    duplicate_zero_shot[lang] = float(data[row1_idx, k])
                duplicate_data['zero_shot'] = duplicate_zero_shot

                # duplicate行的后3个数据是few_shot
                for k, lang in enumerate(languages):
                    duplicate_few_shot[lang] = float(data[row1_idx, k + 3])
                duplicate_data['few_shot'] = duplicate_few_shot

                model_data['duplicate'] = duplicate_data

                # 第二行作为deduplicate
                deduplicate_data = {}
                deduplicate_zero_shot = {}
                deduplicate_few_shot = {}

                # deduplicate行的前3个数据是zero_shot
                for k, lang in enumerate(languages):
                    deduplicate_zero_shot[lang] = float(data[row2_idx, k])
                deduplicate_data['zero_shot'] = deduplicate_zero_shot

                # deduplicate行的后3个数据是few_shot
                for k, lang in enumerate(languages):
                    deduplicate_few_shot[lang] = float(data[row2_idx, k + 3])
                deduplicate_data['few_shot'] = deduplicate_few_shot

                model_data['deduplicate'] = deduplicate_data

                result[model] = model_data

    return result

# 绘制模型性能MAE比较图
def MAE_plot(data_dict, title="Model Performance Comparison (MAE)", figsize=(12, 4), show_labels=True, save_path=None, repeat = True):

    # 提取模型名称
    models = list(data_dict.keys())
    # 自定义模型名称
    if len(models) == 2:
        models_name = ['Qwen7B', 'Qwen7B-ca']
    elif len(models) == 5:
        models_name = ['Qwen', 'DeepSeek', 'GPT', 'Mistral', 'Llama']
    else:
        # 默认情况：使用models中的名称，首字母大写
        models_name = [model.replace('_', ' ').title() for model in models]
        print(f"警告: 检测到 {len(models)} 个模型，使用默认名称: {models_name}")

    # 定义语言和测试类型
    languages = ['en', 'zh', 'ca']
    test_types = ['zero_shot', 'few_shot']

    # 准备数据数组
    data_arrays = {}

    for test_type in test_types:
        for lang in languages:
            key = f"{test_type}_{lang}"
            if repeat == True:
                data_arrays[key] = [data_dict[model]['duplicate'][test_type][lang] for model in models]
            else:
                data_arrays[key] = [data_dict[model]['deduplicate'][test_type][lang] for model in models]

    # 设置图表
    fig, ax = plt.subplots(figsize=figsize)

    # 设置x轴位置
    x = np.arange(len(models))
    width = 0.14  # 柱状图宽度

    # 颜色定义
    colors = {
        'zero_shot_en': '#facea7',#'#FAC795',
        'zero_shot_zh': '#C6CF9D',
        'zero_shot_ca': '#92B4C8',
        'few_shot_en': '#FFE9BE',
        'few_shot_zh': '#D0E0D0',#'#E3EDE0',
        'few_shot_ca': '#ABD3E1'
    }

    # 阴影图案定义（为few-shot添加阴影）
    hatches = {
        'zero_shot_en': '',
        'zero_shot_zh': '',
        'zero_shot_ca': '',
        'few_shot_en': '///',
        'few_shot_zh': '///',
        'few_shot_ca': '///'
    }

    # 绘制柱状图
    bars = []
    bar_labels = []

    for i, (test_type, lang) in enumerate([(tt, lang) for tt in test_types for lang in languages]):
        key = f"{test_type}_{lang}"
        offset = (i - 2.5) * width  # 调整位置
        bar = ax.bar(x + offset, data_arrays[key], width,
                     label=f"{test_type.replace('_', '-').title()} ({lang})",
                     color=colors[key],
                     edgecolor='darkgray',  # 添加灰色边框
                     linewidth=0.3,  # 边框线宽
                     hatch=hatches[key]# 添加阴影图案
                     # alpha=0.2  # 阴影完全不透明
                     )

        bars.append(bar)
        bar_labels.append(f"{test_type.replace('_', '-').title()} ({lang})")

    # 设置标题和标签
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models_name)

    ax.tick_params(axis='both', labelsize=13)  # 同时设置x和y轴刻度标签大小

    # 添加图例
    ax.legend(loc='upper right', fontsize=11, ncol=2)

    # 修改这里：固定y轴范围为[0,5]
    ax.set_ylim(0, 3)

    # 添加数据标签
    if show_labels:
        # 获取y轴的最大值，用于计算合适的偏移量
        y_max = ax.get_ylim()[1]
        # 设置偏移量为y轴最大值的2%，这样偏移量会根据数据范围自适应
        offset = y_max * 0.02
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                # 在原始高度基础上增加偏移量，使标签离柱状图更远
                label_y = height + offset
                ax.text(bar.get_x() + bar.get_width() / 2., label_y,
                        f'{height:.4f}', ha='center', va='bottom',
                        fontsize=10, rotation=90, color='gray')

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    # 显示图表
    plt.show()

    return fig, ax

# 绘制模型性能MAE比较图(version2)
def MAE_plot2(data_dict, title="Model Performance Comparison (MAE)", figsize=(12, 4), show_labels=True, save_path=None, repeat = True):
    # 提取模型名称
    models = list(data_dict.keys())
    # 自定义模型名称
    if len(models) == 2:
        models_name = ['Qwen7B', 'CantoLLM7B']
    elif len(models) == 5:
        models_name = ['Qwen', 'DeepSeek', 'GPT', 'Mistral', 'Llama']
    else:
        # 默认情况：使用models中的名称，首字母大写
        models_name = [model.replace('_', ' ').title() for model in models]
        print(f"警告: 检测到 {len(models)} 个模型，使用默认名称: {models_name}")

    # 定义语言和测试类型
    languages = ['en', 'zh', 'ca']
    test_types = ['zero_shot', 'few_shot']

    # 准备数据数组
    data_arrays = {}

    for test_type in test_types:
        for lang in languages:
            key = f"{test_type}_{lang}"
            if repeat == True:
                data_arrays[key] = [data_dict[model]['duplicate'][test_type][lang] for model in models]
            else:
                data_arrays[key] = [data_dict[model]['deduplicate'][test_type][lang] for model in models]

    # 设置图表
    fig, ax = plt.subplots(figsize=figsize)

    # 设置x轴位置
    x = np.arange(len(models))
    width = 0.14  # 柱状图宽度 #0.14

    # 颜色定义
    colors = {
        'zero_shot_en': '#E69F00',#'#FAC795',
        'zero_shot_zh': '#7A9E3A',
        'zero_shot_ca': '#3B6FA5',
        'few_shot_en': '#E69F00',
        'few_shot_zh': '#7A9E3A',#'#E3EDE0',
        'few_shot_ca': '#3B6FA5'
    }

    # 绘制柱状图
    bars = []
    bar_labels = []

    for i, (test_type, lang) in enumerate([(tt, lang) for tt in test_types for lang in languages]):
        key = f"{test_type}_{lang}"
        offset = (i - 2.5) * width

        if test_type == 'zero_shot':
            # ===== zero-shot：空心柱 =====
            bar = ax.bar(
                x + offset,
                data_arrays[key],
                width,
                label=f"{test_type.replace('_', '-').title()} ({lang})",
                facecolor='none',  # 空心
                edgecolor=colors[key],  # 边框用原颜色
                linewidth=1.2
            )
        else:
            # ===== few-shot：实心柱 =====
            bar = ax.bar(
                x + offset,
                data_arrays[key],
                width,
                label=f"{test_type.replace('_', '-').title()} ({lang})",
                color=colors[key],  # 实心填充
                edgecolor='none',
                linewidth=0.5
            )

        bars.append(bar)
        bar_labels.append(f"{test_type.replace('_', '-').title()} ({lang})")

    # 设置标题和标签
    ax.set_xlabel(metric, fontsize=15)
    ax.set_ylabel('Mean Absolute Error', fontsize=15)
    # ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_name)

    ax.tick_params(axis='both', labelsize=14)  # 同时设置x和y轴刻度标签大小

    # 修改这里：固定y轴范围为[0,5]
    ax.set_ylim(0, 4)

    # 添加数据标签
    if show_labels:
        # 获取y轴的最大值，用于计算合适的偏移量
        y_max = ax.get_ylim()[1]
        # 设置偏移量为y轴最大值的2%，这样偏移量会根据数据范围自适应
        offset = y_max * 0.02
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                # 在原始高度基础上增加偏移量，使标签离柱状图更远
                label_y = height + offset
                ax.text(bar.get_x() + bar.get_width() / 2., label_y,
                        f'{height:.4f}', ha='center', va='bottom',
                        fontsize=10, rotation=90, color='gray')

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.axvline(x=0.5, color='gray', linewidth=1, linestyle='--', zorder=0, alpha=0.6)
    # ax.axvline(x=1.5, color='gray', linewidth=1, linestyle='--', zorder=0, alpha=0.6)
    # ax.axvline(x=2.5, color='gray', linewidth=1, linestyle='--', zorder=0, alpha=0.6)
    # ax.axvline(x=3.5, color='gray', linewidth=1, linestyle='--', zorder=0, alpha=0.6)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    # 显示图表
    plt.show()

    return fig, ax

# 读取Spearman Excel文件并构建数据字典
def read_SPEARMANexcel_to_dict(file_path, language, metric, model_number):

    def parse_number(value):
        """解析科学记数法的字符串为浮点数"""
        if pd.isna(value):
            return 0.0

        str_val = str(value).strip()

        # 处理E+00, E-00等格式
        if 'E' in str_val.upper():
            try:
                # 标准化科学记数法格式
                str_val = str_val.replace('E+', 'e+').replace('E-', 'e-').replace('E', 'e')
                return float(str_val)
            except:
                return 0.0
        else:
            try:
                return float(str_val)
            except:
                return 0.0

    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name='Spearman', header=None)

    if model_number == 5:
        # 定义模型名称顺序
        models = ['qwen', 'deepseek', 'gpt', 'mistral', 'llama']
    elif model_number == 2:
        models = ['qwen7b', 'qwen7b_ca']
    else:
        pass

    languages = ['en', 'zh', 'ca']

    # 初始化结果字典
    result = {}

    # 根据语言和指标查找数据
    language_found = False
    metric_found = False
    start_row = 0

    # 遍历查找语言和指标位置
    for i in range(len(df)):
        cell_value = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""

        # 查找语言
        if language in cell_value and not language_found:
            language_found = True
            continue

        # 查找指标
        if language_found and metric in cell_value and not metric_found:
            metric_found = True
            start_row = i + 1  # 跳过表头行
            break

    if not metric_found:
        raise ValueError(f"未找到 {language} 的 {metric} 数据")

    # 读取数据
    if language == 'Cantonese':
        # 粤语有5行数据，每行有12个值（6对spearman和p_value）
        data_rows = 5
        data = df.iloc[start_row:start_row + data_rows, :12].values

        for i, model in enumerate(models):
            if i < len(data):
                model_data = {}

                # 创建duplicate层
                duplicate_data = {}

                # duplicate层的zero_shot部分
                zero_shot_data = {}
                for k, lang in enumerate(languages):
                    # 每对数据：列0-1对应en，列2-3对应zh，列4-5对应ca
                    col_idx = k * 2
                    zero_shot_data[lang] = {
                        'spearman': parse_number(data[i, col_idx]),
                        'p_value': parse_number(data[i, col_idx + 1])
                    }
                duplicate_data['zero_shot'] = zero_shot_data

                # duplicate层的few_shot部分
                few_shot_data = {}
                for k, lang in enumerate(languages):
                    # 后3对数据：列6-7对应en，列8-9对应zh，列10-11对应ca
                    col_idx = 6 + k * 2
                    few_shot_data[lang] = {
                        'spearman': parse_number(data[i, col_idx]),
                        'p_value': parse_number(data[i, col_idx + 1])
                    }
                duplicate_data['few_shot'] = few_shot_data

                model_data['duplicate'] = duplicate_data
                model_data['deduplicate'] = {}  # 粤语没有deduplicate数据，设为空字典

                result[model] = model_data

    elif language == 'Mandarin':
        # 普通话有10行数据，每两行对应一个模型（第一行duplicate，第二行deduplicate）
        data_rows = 10
        data = df.iloc[start_row:start_row + data_rows, :12].values

        # 每两行对应一个模型
        for i, model in enumerate(models):
            row1_idx = i * 2  # duplicate行
            row2_idx = i * 2 + 1  # deduplicate行

            model_data = {}

            if row1_idx < data_rows:
                # 创建duplicate层
                duplicate_data = {}

                # duplicate层的zero_shot部分
                duplicate_zero_shot = {}
                for k, lang in enumerate(languages):
                    col_idx = k * 2
                    duplicate_zero_shot[lang] = {
                        'spearman': parse_number(data[row1_idx, col_idx]),
                        'p_value': parse_number(data[row1_idx, col_idx + 1])
                    }
                duplicate_data['zero_shot'] = duplicate_zero_shot

                # duplicate层的few_shot部分
                duplicate_few_shot = {}
                for k, lang in enumerate(languages):
                    col_idx = 6 + k * 2
                    duplicate_few_shot[lang] = {
                        'spearman': parse_number(data[row1_idx, col_idx]),
                        'p_value': parse_number(data[row1_idx, col_idx + 1])
                    }
                duplicate_data['few_shot'] = duplicate_few_shot

                model_data['duplicate'] = duplicate_data

            if row2_idx < data_rows:
                # 创建deduplicate层
                deduplicate_data = {}

                # deduplicate层的zero_shot部分
                deduplicate_zero_shot = {}
                for k, lang in enumerate(languages):
                    col_idx = k * 2
                    deduplicate_zero_shot[lang] = {
                        'spearman': parse_number(data[row2_idx, col_idx]),
                        'p_value': parse_number(data[row2_idx, col_idx + 1])
                    }
                deduplicate_data['zero_shot'] = deduplicate_zero_shot

                # deduplicate层的few_shot部分
                deduplicate_few_shot = {}
                for k, lang in enumerate(languages):
                    col_idx = 6 + k * 2
                    deduplicate_few_shot[lang] = {
                        'spearman': parse_number(data[row2_idx, col_idx]),
                        'p_value': parse_number(data[row2_idx, col_idx + 1])
                    }
                deduplicate_data['few_shot'] = deduplicate_few_shot

                model_data['deduplicate'] = deduplicate_data

            result[model] = model_data

    return result

# 绘制Spearman相关系数比较图
def Spearman_plot(data_dict, figsize=(18, 6), sharey=True, show_labels=True, save_path=None, title="Spearman Correlation Comparison Across Languages"):

        # 从数据中提取模型名称和语言类型
        models = list(data_dict.keys())

        if len(models) == 2:
            models_name = ['Qwen7B', 'Qwen7B-ca']
        elif len(models) == 5:
            models_name = ['Qwen', 'DeepSeek', 'GPT', 'Mistral', 'Llama']
        else:
            # 默认情况：使用models中的名称，首字母大写
            models_name = [model.replace('_', ' ').title() for model in models]
            print(f"警告: 检测到 {len(models)} 个模型，使用默认名称: {models_name}")

        # 从第一个模型的数据中获取所有语言和测试类型
        first_model = data_dict[models[0]]
        test_types = list(first_model.keys())  # 如 ['zero_shot', 'few_shot']

        # 获取语言列表（从第一个测试类型中）
        first_test_type = test_types[0]
        languages = list(first_model[first_test_type].keys())  # 如 ['en', 'zh', 'ca']

        # 颜色定义
        colors = {
            'zero_shot': '#4C72B0',
            'few_shot': '#55A868'
        }

        # 绘制综合图表（所有语言和模型）
        fig, axes = plt.subplots(1, len(languages), figsize=figsize, sharey=sharey)

        # 如果只有一个语言，axes不是数组，需要转换为列表以便统一处理
        if len(languages) == 1:
            axes = [axes]

        for idx, lang in enumerate(languages):
            ax = axes[idx]

            x = np.arange(len(models))
            width = 0.35

            for i, test_type in enumerate(test_types):
                # 提取该语言下所有模型的spearman值
                spearman_values = []
                for model in models:
                    try:
                        spearman_values.append(data_dict[model][test_type][lang]['spearman'])
                    except KeyError:
                        # 如果数据缺少某些键，用0填充
                        spearman_values.append(0)

                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, spearman_values, width,
                              label=test_type.replace('_', ' ').title(),
                              color=colors.get(test_type, f'C{i}'),  # 使用默认颜色如果未指定
                              edgecolor='black',
                              linewidth=0.5)

                # 标注数值
                if show_labels:
                    for bar, spearman in zip(bars, spearman_values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{spearman:.3f}', ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Models', fontsize=12)
            ax.set_title(f'{lang.upper()}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models_name, fontsize=11)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.0)

            if idx == 0:
                ax.set_ylabel('Spearman Correlation', fontsize=12)

            # 只在第一个子图上显示图例
            if idx == 0:
                ax.legend(loc='upper right', fontsize=10)

        # 添加整体标题
        # plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局为整体标题留空间

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        # 显示图表
        plt.show()

        return fig, axes

# 绘制Spearman相关系数热力图
def Spearman_heatmap2(data_dict, figsize=(15, 10), cmap='YlGn', show_values=True,
                      save_path=None, title="Spearman Correlation Heatmap",
                      include_pvalue=False, pvalue_threshold=0.05,
                      row_labels_type='combined',
                      text_color='black', text_size=12,
                      highlight_color='grey', repeat=True):

    # 从数据中提取模型名称
    models = list(data_dict.keys())
    if len(models) == 2:
        models_name = ['Qwen7B', 'Qwen7B-ca']
    elif len(models) == 5:
        models_name = ['Qwen', 'DeepSeek', 'GPT', 'Mistral', 'Llama']
    else:
        # 默认情况：使用models中的名称，首字母大写
        models_name = [model.replace('_', ' ').title() for model in models]
        print(f"警告: 检测到 {len(models)} 个模型，使用默认名称: {models_name}")

    # 定义测试类型和语言
    test_types = ['zero_shot', 'few_shot']
    languages = ['en', 'zh', 'ca']

    # 根据 repeat 参数确定使用哪个数据层
    data_layer = 'duplicate' if repeat else 'deduplicate'

    # 创建数据矩阵
    rows = []
    row_labels = []
    pvalue_matrix = []

    # 构建行标签和矩阵数据
    if row_labels_type == 'combined':
        # 组合显示：测试类型 + 语言
        for test_type in test_types:
            for lang in languages:
                row = []
                pvalue_row = []
                for model in models:
                    try:
                        # 根据 data_layer 访问数据
                        row.append(data_dict[model][data_layer][test_type][lang]['spearman'])
                        pvalue_row.append(data_dict[model][data_layer][test_type][lang]['p_value'])
                    except KeyError:
                        row.append(0)
                        pvalue_row.append(1.0)
                rows.append(row)
                pvalue_matrix.append(pvalue_row)
                row_labels.append(f"{test_type.replace('_', '-').title()}\n({lang})")
    else:
        # 分开显示：先显示测试类型，再显示语言
        for test_type in test_types:
            # 先添加测试类型行
            row = []
            pvalue_row = []
            for model in models:
                row.append(np.nan)
                pvalue_row.append(1.0)
            rows.append(row)
            pvalue_matrix.append(pvalue_row)
            row_labels.append(f"{test_type.replace('_', '-').title()}")

            # 再添加该测试类型下的各个语言
            for lang in languages:
                row = []
                pvalue_row = []
                for model in models:
                    try:
                        # 根据 data_layer 访问数据
                        row.append(data_dict[model][data_layer][test_type][lang]['spearman'])
                        pvalue_row.append(data_dict[model][data_layer][test_type][lang]['p_value'])
                    except KeyError:
                        row.append(0)
                        pvalue_row.append(1.0)
                rows.append(row)
                pvalue_matrix.append(pvalue_row)
                row_labels.append(f"{lang}")

    # 转换为numpy数组
    data_matrix = np.array(rows)

    # 创建自定义 Pastel2 色彩映射，包含10种颜色
    def create_custom_pastel2_10():
        """
        创建包含10种颜色的 Pastel2 色彩映射
        颜色选择策略：从 Pastel2 中选取所有8种颜色，再从 Set3 中选取2种互补颜色
        """
        # 获取原始 Pastel2 的8种颜色
        pastel2_all = plt.cm.Pastel2.colors

        # 获取 Set3 的12种颜色，用于补充
        set3_all = plt.cm.Set3.colors

        # 从 Pastel2 中选取所有8种颜色
        selected_colors = list(pastel2_all)

        # 从 Set3 中选取2种颜色作为补充，选择与 Pastel2 风格协调的颜色
        # 选择 Set3 中的索引 0, 3 的颜色，这些颜色与 Pastel2 风格相似
        selected_colors.append('#FFD7D6')
        selected_colors.append('#ADD8E6')

        # print(selected_colors)

        # 创建新的 ListedColormap
        custom_cmap = ListedColormap(selected_colors, name='Pastel2_10')

        return custom_cmap, selected_colors

    # 创建自定义 Pastel2 色彩映射（10种颜色）
    pastel2_10, colors_10 = create_custom_pastel2_10()

    def create_custom_pastel1(n_colors=5):
        """
        创建只包含 Pastel1 前 n_colors 种颜色的新色彩映射
        """
        # 获取原始 Pastel1 颜色
        pastel1_all = plt.cm.Pastel1.colors

        # 检查是否有足够的颜色
        if n_colors > len(pastel1_all):
            print(f"警告: Pastel1 只有 {len(pastel1_all)} 种颜色，将使用所有颜色")
            n_colors = len(pastel1_all)

        # 选取前 n_colors 种颜色
        selected_colors = pastel1_all[:n_colors]

        # 创建新的 ListedColormap
        custom_cmap = ListedColormap(selected_colors, name=f'Pastel1_{n_colors}')

        return custom_cmap, selected_colors

    # 创建只包含5种颜色的 Pastel1 色彩映射
    pastel1_5, colors_5 = create_custom_pastel1(5)

    def create_custom_pastel2(n_colors=5):
        """
        创建只包含 Pastel2 前 n_colors 种颜色的新色彩映射
        """
        # 获取原始 Pastel1 颜色
        pastel2_all = plt.cm.Pastel2.colors

        # 检查是否有足够的颜色
        if n_colors > len(pastel2_all):
            print(f"警告: Pastel2 只有 {len(pastel2_all)} 种颜色，将使用所有颜色")
            n_colors = len(pastel2_all)

        # 选取前 n_colors 种颜色
        selected_colors = (pastel2_all[1],pastel2_all[4],pastel2_all[5],pastel2_all[2],pastel2_all[6],)
        # print(selected_colors)

        # 创建新的 ListedColormap
        custom_cmap = ListedColormap(selected_colors, name=f'Pastel2_{n_colors}')

        return custom_cmap, selected_colors

    # 创建只包含5种颜色的 Pastel1 色彩映射
    pastel2_5, colors_5 = create_custom_pastel2(5)

    # 创建自定义颜色方案
    def get_cmap(cmap_name):
        """获取颜色映射"""
        cmap_options = {
            'YlOrRd': plt.cm.YlOrRd,  # 黄橙红，适合相关性热力图
            'viridis': plt.cm.viridis,  # 现代科学可视化常用
            'plasma': plt.cm.plasma,  # 高对比度
            'coolwarm': plt.cm.coolwarm,  # 冷暖色对比
            'RdBu_r': plt.cm.RdBu_r,  # 红蓝反转，高值红色
            'Blues': plt.cm.Blues,  # 蓝色渐变
            'Greens': plt.cm.Greens,  # 绿色渐变
            'Pastel1': plt.cm.Pastel1,  # 柔和颜色
            'Pastel2': plt.cm.Pastel2,  # 柔和颜色
            'Spectral': plt.cm.Spectral,  # 彩虹色
            'PuBuGn': plt.cm.PuBuGn,  # 紫蓝绿
            'Oranges': plt.cm.Oranges,  # 橙色渐变
            'Purples': plt.cm.Purples,  # 紫色渐变
            'YlGn': plt.cm.YlGn,  # 黄绿渐变
            'Pastel1_5': pastel1_5, #自定义pastel1
            'Pastel2_5': pastel2_5, #自定义pastel2（5种颜色）
            'Pastel2_10': pastel2_10,  # 自定义pastel2（10种颜色）
        }
        return cmap_options.get(cmap_name, plt.cm.Pastel1)

    # 自动选择文本颜色的函数
    def get_text_color(cell_value, cmap, vmin=0, vmax=1):
        """根据单元格背景颜色自动选择文本颜色"""
        if text_color == 'auto':
            # 根据单元格值计算颜色亮度
            norm_value = (cell_value - vmin) / (vmax - vmin)
            rgba = cmap(norm_value)
            # 计算亮度 (YIQ颜色空间)
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            return 'black' if brightness < 0.5 else 'black'
        else:
            return text_color

    # 获取颜色映射
    selected_cmap = get_cmap(cmap)

    # 创建热力图
    fig, ax = plt.subplots(figsize=figsize)

    # 判断是否使用离散色彩映射（10种颜色）
    # 如果是自定义的10种颜色映射，我们使用离散的颜色边界
    if cmap in ['Pastel2_10']:
        # 创建离散颜色边界：10个区间，每个区间0.1
        from matplotlib.colors import BoundaryNorm

        # 定义边界：从0到1，步长为0.1，总共11个边界值
        bounds = np.arange(0, 1.1, 0.1)

        # 创建BoundaryNorm，将数据映射到离散的颜色区间
        norm = BoundaryNorm(bounds, selected_cmap.N)

        # 对于分开显示，我们需要处理NaN值
        if row_labels_type == 'separate':
            mask = np.isnan(data_matrix)
            heatmap = sns.heatmap(data_matrix,
                                  annot=show_values,
                                  fmt='.3f' if show_values else '',
                                  cmap=selected_cmap,
                                  norm=norm,  # 使用离散的norm
                                  linewidths=0.5,
                                  linecolor='gray',
                                  mask=mask,
                                  cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8,
                                            'ticks': bounds[:-1] + 0.05,  # 设置刻度在区间中间
                                            'boundaries': bounds},  # 设置边界
                                  annot_kws={'size': text_size, 'color': text_color} if show_values else None,
                                  ax=ax)
        else:
            heatmap = sns.heatmap(data_matrix,
                                  annot=show_values,
                                  fmt='.3f' if show_values else '',
                                  cmap=selected_cmap,
                                  norm=norm,  # 使用离散的norm
                                  linewidths=0.5,
                                  linecolor='gray',
                                  cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8,
                                            'ticks': bounds[:-1] + 0.05,  # 设置刻度在区间中间
                                            'boundaries': bounds},  # 设置边界
                                  annot_kws={'size': text_size, 'color': text_color} if show_values else None,
                                  ax=ax)
    else:
        # 对于分开显示，我们需要处理NaN值
        if row_labels_type == 'separate':
            mask = np.isnan(data_matrix)
            heatmap = sns.heatmap(data_matrix,
                                  annot=show_values,
                                  fmt='.3f' if show_values else '',
                                  cmap=selected_cmap,
                                  vmin=0,
                                  vmax=1,
                                  linewidths=0.5,
                                  linecolor='gray',
                                  mask=mask,
                                  cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8},
                                  annot_kws={'size': text_size, 'color': text_color} if show_values else None,
                                  ax=ax)
        else:
            heatmap = sns.heatmap(data_matrix,
                                  annot=show_values,
                                  fmt='.3f' if show_values else '',
                                  cmap=selected_cmap,
                                  vmin=0,
                                  vmax=1,
                                  linewidths=0.5,
                                  linecolor='gray',
                                  cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8},
                                  annot_kws={'size': text_size, 'color': text_color} if show_values else None,
                                  ax=ax)

    # 如果使用自动文本颜色，需要单独设置每个单元格的文本颜色
    if show_values and text_color == 'auto':
        # 遍历所有文本框，根据背景色设置文本颜色
        for i, text_row in enumerate(heatmap.texts):
            # 计算文本对应的矩阵位置
            n_rows, n_cols = data_matrix.shape
            # 热力图文本是按行顺序排列的
            row = i // n_cols
            col = i % n_cols
            # 跳过被掩码的单元格
            if row_labels_type == 'separate' and np.isnan(data_matrix[row, col]):
                continue

            cell_value = data_matrix[row, col]
            # 获取文本颜色
            txt_color = get_text_color(cell_value, selected_cmap)
            text_row.set_color(txt_color)
            text_row.set_fontweight('bold')

    # 自定义标签
    ax.set_xlabel(f"{metric}", fontsize=12)# fontweight='bold'
    ax.set_ylabel('Prompt Types', fontsize=12)# fontweight='bold'

    # 设置x轴标签（模型）
    ax.set_xticks(np.arange(len(models)) + 0.5)
    ax.set_xticklabels(models_name, fontsize=12, rotation=0)# fontweight='bold',

    # 设置y轴标签（测试类型 + 语言）
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    yticklabels = []
    for label in row_labels:
        # 为分隔行添加粗体
        if row_labels_type == 'separate' and label.endswith(')'):
            yticklabels.append(label)
        else:
            yticklabels.append(label)
    ax.set_yticklabels(yticklabels, fontsize=11)#, fontweight='bold'

    # 添加显著性标记（如果需要）
    if include_pvalue and show_values:
        for i in range(data_matrix.shape[0]):
            for j in range(data_matrix.shape[1]):
                # 跳过NaN值（对于分开显示模式）
                if row_labels_type == 'separate' and np.isnan(data_matrix[i, j]):
                    continue

                p_value = pvalue_matrix[i][j]
                if isinstance(p_value, (int, float)) and p_value < pvalue_threshold:
                    # 根据显著性水平添加不同数量的星号
                    if p_value < 0.001:
                        star = '***'
                    elif p_value < 0.01:
                        star = '**'
                    elif p_value < pvalue_threshold:
                        star = '*'
                    else:
                        star = ''

                    if star:
                        # 在单元格中添加星号
                        ax.text(j + 0.5, i + 0.7, star,
                                ha='center', va='center',
                                fontsize=text_size, fontweight='bold',
                                color=highlight_color)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存至: {save_path}")

    # 显示图表
    plt.show()

    return fig, ax

# 读取t-test Excel文件并构建数据字典
def read_TTESTexcel_to_dict(file_path, language, metric):

    def parse_number(value):
        """解析科学记数法的字符串为浮点数"""
        if pd.isna(value):
            return 0.0

        str_val = str(value).strip()

        # 处理E+00, E-00等格式
        if 'E' in str_val.upper():
            try:
                # 标准化科学记数法格式
                str_val = str_val.replace('E+', 'e+').replace('E-', 'e-').replace('E', 'e')
                return float(str_val)
            except:
                return 0.0
        else:
            try:
                return float(str_val)
            except:
                return 0.0

    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name='Paired-T test', header=None)

    # 定义模型名称顺序
    # models = ['qwen', 'deepseek', 'gpt', 'mistral', 'llama']
    models = ['qwen7b', 'qwen7b_ca']

    languages = ['en', 'zh', 'ca']

    # 初始化结果字典
    result = {}

    # 根据语言和指标查找数据
    language_found = False
    metric_found = False
    start_row = 0

    # 遍历查找语言和指标位置
    for i in range(len(df)):
        cell_value = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""

        # 查找语言
        if language in cell_value and not language_found:
            language_found = True
            continue

        # 查找指标
        if language_found and metric in cell_value and not metric_found:
            metric_found = True
            start_row = i + 1  # 跳过表头行
            break

    if not metric_found:
        raise ValueError(f"未找到 {language} 的 {metric} 数据")

    # 读取数据
    if language == 'Cantonese':
        # 粤语有5行数据，每行有12个值（6对spearman和p_value）
        data_rows = 5
        data = df.iloc[start_row:start_row + data_rows, :12].values

        for i, model in enumerate(models):
            if i < len(data):
                model_data = {}

                # 创建duplicate层
                duplicate_data = {}

                # duplicate层的zero_shot部分
                zero_shot_data = {}
                for k, lang in enumerate(languages):
                    # 每对数据：列0-1对应en，列2-3对应zh，列4-5对应ca
                    col_idx = k * 2
                    zero_shot_data[lang] = {
                        'spearman': parse_number(data[i, col_idx]),
                        'p_value': parse_number(data[i, col_idx + 1])
                    }
                duplicate_data['zero_shot'] = zero_shot_data

                # duplicate层的few_shot部分
                few_shot_data = {}
                for k, lang in enumerate(languages):
                    # 后3对数据：列6-7对应en，列8-9对应zh，列10-11对应ca
                    col_idx = 6 + k * 2
                    few_shot_data[lang] = {
                        'spearman': parse_number(data[i, col_idx]),
                        'p_value': parse_number(data[i, col_idx + 1])
                    }
                duplicate_data['few_shot'] = few_shot_data

                model_data['duplicate'] = duplicate_data
                model_data['deduplicate'] = {}  # 粤语没有deduplicate数据，设为空字典

                result[model] = model_data

    elif language == 'Mandarin':
        # 普通话有10行数据，每两行对应一个模型（第一行duplicate，第二行deduplicate）
        data_rows = 10
        data = df.iloc[start_row:start_row + data_rows, :12].values

        # 每两行对应一个模型
        for i, model in enumerate(models):
            row1_idx = i * 2  # duplicate行
            row2_idx = i * 2 + 1  # deduplicate行

            model_data = {}

            if row1_idx < data_rows:
                # 创建duplicate层
                duplicate_data = {}

                # duplicate层的zero_shot部分
                duplicate_zero_shot = {}
                for k, lang in enumerate(languages):
                    col_idx = k * 2
                    duplicate_zero_shot[lang] = {
                        'spearman': parse_number(data[row1_idx, col_idx]),
                        'p_value': parse_number(data[row1_idx, col_idx + 1])
                    }
                duplicate_data['zero_shot'] = duplicate_zero_shot

                # duplicate层的few_shot部分
                duplicate_few_shot = {}
                for k, lang in enumerate(languages):
                    col_idx = 6 + k * 2
                    duplicate_few_shot[lang] = {
                        'spearman': parse_number(data[row1_idx, col_idx]),
                        'p_value': parse_number(data[row1_idx, col_idx + 1])
                    }
                duplicate_data['few_shot'] = duplicate_few_shot

                model_data['duplicate'] = duplicate_data

            if row2_idx < data_rows:
                # 创建deduplicate层
                deduplicate_data = {}

                # deduplicate层的zero_shot部分
                deduplicate_zero_shot = {}
                for k, lang in enumerate(languages):
                    col_idx = k * 2
                    deduplicate_zero_shot[lang] = {
                        'spearman': parse_number(data[row2_idx, col_idx]),
                        'p_value': parse_number(data[row2_idx, col_idx + 1])
                    }
                deduplicate_data['zero_shot'] = deduplicate_zero_shot

                # deduplicate层的few_shot部分
                deduplicate_few_shot = {}
                for k, lang in enumerate(languages):
                    col_idx = 6 + k * 2
                    deduplicate_few_shot[lang] = {
                        'spearman': parse_number(data[row2_idx, col_idx]),
                        'p_value': parse_number(data[row2_idx, col_idx + 1])
                    }
                deduplicate_data['few_shot'] = deduplicate_few_shot

                model_data['deduplicate'] = deduplicate_data

            result[model] = model_data

    return result

# 绘制t-test相关系数热力图
def Ttest_heatmap(data_dict, figsize=(15, 10), cmap='YlGn', show_values=True,
                      save_path=None, title="Spearman Correlation Heatmap",
                      include_pvalue=False, pvalue_threshold=0.05,
                      row_labels_type='combined',
                      text_color='black', text_size=12,
                      highlight_color='darkgrey', repeat=True):

    # 从数据中提取模型名称
    models = list(data_dict.keys())
    if len(models) == 2:
        models_name = ['Qwen7B', 'Qwen7B-ca']
    elif len(models) == 5:
        models_name = ['Qwen', 'DeepSeek', 'GPT', 'Mistral', 'Llama']
    else:
        # 默认情况：使用models中的名称，首字母大写
        models_name = [model.replace('_', ' ').title() for model in models]
        print(f"警告: 检测到 {len(models)} 个模型，使用默认名称: {models_name}")

    # 定义测试类型和语言
    test_types = ['zero_shot', 'few_shot']
    languages = ['en', 'zh', 'ca']

    # 根据 repeat 参数确定使用哪个数据层
    data_layer = 'duplicate' if repeat else 'deduplicate'

    # 创建数据矩阵
    rows = []
    row_labels = []
    pvalue_matrix = []

    # 构建行标签和矩阵数据
    if row_labels_type == 'combined':
        # 组合显示：测试类型 + 语言
        for test_type in test_types:
            for lang in languages:
                row = []
                pvalue_row = []
                for model in models:
                    try:
                        # 根据 data_layer 访问数据
                        row.append(data_dict[model][data_layer][test_type][lang]['spearman'])
                        pvalue_row.append(data_dict[model][data_layer][test_type][lang]['p_value'])
                    except KeyError:
                        row.append(0)
                        pvalue_row.append(1.0)
                rows.append(row)
                pvalue_matrix.append(pvalue_row)
                row_labels.append(f"{test_type.replace('_', '-').title()}\n({lang})")
    else:
        # 分开显示：先显示测试类型，再显示语言
        for test_type in test_types:
            # 先添加测试类型行
            row = []
            pvalue_row = []
            for model in models:
                row.append(np.nan)
                pvalue_row.append(1.0)
            rows.append(row)
            pvalue_matrix.append(pvalue_row)
            row_labels.append(f"{test_type.replace('_', '-').title()}")

            # 再添加该测试类型下的各个语言
            for lang in languages:
                row = []
                pvalue_row = []
                for model in models:
                    try:
                        # 根据 data_layer 访问数据
                        row.append(data_dict[model][data_layer][test_type][lang]['spearman'])
                        pvalue_row.append(data_dict[model][data_layer][test_type][lang]['p_value'])
                    except KeyError:
                        row.append(0)
                        pvalue_row.append(1.0)
                rows.append(row)
                pvalue_matrix.append(pvalue_row)
                row_labels.append(f"{lang}")

    # 转换为numpy数组
    data_matrix = np.array(rows)
    valid_data = data_matrix[~np.isnan(data_matrix)]

    vmin = -200
    vmax = 200

    # 创建自定义 Pastel2 色彩映射，包含10种颜色
    def create_custom_pastel2_10():
        """
        创建包含10种颜色的 Pastel2 色彩映射
        颜色选择策略：从 Pastel2 中选取所有8种颜色，再从 Set3 中选取2种互补颜色
        """
        # 获取原始 Pastel2 的8种颜色
        pastel2_all = plt.cm.Pastel2.colors

        # 获取 Set3 的12种颜色，用于补充
        set3_all = plt.cm.Set3.colors

        # 从 Pastel2 中选取所有8种颜色
        selected_colors = list(pastel2_all)

        # 从 Set3 中选取2种颜色作为补充，选择与 Pastel2 风格协调的颜色
        # 选择 Set3 中的索引 0, 3 的颜色，这些颜色与 Pastel2 风格相似
        selected_colors.append('#FFD7D6')
        selected_colors.append('#ADD8E6')

        # print(selected_colors)

        # 创建新的 ListedColormap
        custom_cmap = ListedColormap(selected_colors, name='Pastel2_10')

        return custom_cmap, selected_colors

    # 创建自定义 Pastel2 色彩映射（10种颜色）
    pastel2_10, colors_10 = create_custom_pastel2_10()

    def create_custom_pastel1(n_colors=5):
        """
        创建只包含 Pastel1 前 n_colors 种颜色的新色彩映射
        """
        # 获取原始 Pastel1 颜色
        pastel1_all = plt.cm.Pastel1.colors

        # 检查是否有足够的颜色
        if n_colors > len(pastel1_all):
            print(f"警告: Pastel1 只有 {len(pastel1_all)} 种颜色，将使用所有颜色")
            n_colors = len(pastel1_all)

        # 选取前 n_colors 种颜色
        selected_colors = pastel1_all[:n_colors]

        # 创建新的 ListedColormap
        custom_cmap = ListedColormap(selected_colors, name=f'Pastel1_{n_colors}')

        return custom_cmap, selected_colors

    # 创建只包含5种颜色的 Pastel1 色彩映射
    pastel1_5, colors_5 = create_custom_pastel1(5)

    def create_custom_pastel2(n_colors=5):
        """
        创建只包含 Pastel2 前 n_colors 种颜色的新色彩映射
        """
        # 获取原始 Pastel1 颜色
        pastel2_all = plt.cm.Pastel2.colors

        # 检查是否有足够的颜色
        if n_colors > len(pastel2_all):
            print(f"警告: Pastel2 只有 {len(pastel2_all)} 种颜色，将使用所有颜色")
            n_colors = len(pastel2_all)

        # 选取前 n_colors 种颜色
        selected_colors = (pastel2_all[1],pastel2_all[4],pastel2_all[5],pastel2_all[2],pastel2_all[6],)
        # print(selected_colors)

        # 创建新的 ListedColormap
        custom_cmap = ListedColormap(selected_colors, name=f'Pastel2_{n_colors}')

        return custom_cmap, selected_colors

    # 创建只包含5种颜色的 Pastel1 色彩映射
    pastel2_5, colors_5 = create_custom_pastel2(5)

    def create_pastel_coolwarm():
        """创建柔和的coolwarm配色"""
        colors = [
            (0.1, 0.4, 0.85, 1.0),  # 加深的蓝色 - 保持最深蓝
            (0.2, 0.5, 0.88, 1.0),  # 中等蓝色 - 更深更鲜
            (0.55, 0.68, 0.85, 1.0),  # 淡蓝色 - 更浓更深
            (0.96, 0.96, 0.96, 1.0),  # 几乎白色 - 保持原样
            (0.95, 0.8, 0.7, 1.0),  # 淡红色 - 更浓郁
            (0.95, 0.6, 0.48, 1.0),  # 中等红色 - 更深更饱和
            (0.85, 0.45, 0.25, 1.0)  # 加深的红色 - 保持最深红
        ]
        return LinearSegmentedColormap.from_list('pastel_coolwarm', colors, N=256)

    # 创建自定义颜色方案
    def get_cmap(cmap_name):
        """获取颜色映射"""
        cmap_options = {
            'YlOrRd': plt.cm.YlOrRd,  # 黄橙红，适合相关性热力图
            'viridis': plt.cm.viridis,  # 现代科学可视化常用
            'plasma': plt.cm.plasma,  # 高对比度
            'coolwarm': plt.cm.coolwarm,  # 冷暖色对比
            'RdBu_r': plt.cm.RdBu_r,  # 红蓝反转，高值红色
            'Blues': plt.cm.Blues,  # 蓝色渐变
            'Greens': plt.cm.Greens,  # 绿色渐变
            'Pastel1': plt.cm.Pastel1,  # 柔和颜色
            'Pastel2': plt.cm.Pastel2,  # 柔和颜色
            'Spectral': plt.cm.Spectral,  # 彩虹色
            'PuBuGn': plt.cm.PuBuGn,  # 紫蓝绿
            'Oranges': plt.cm.Oranges,  # 橙色渐变
            'Purples': plt.cm.Purples,  # 紫色渐变
            'YlGn': plt.cm.YlGn,  # 黄绿渐变
            'Pastel1_5': pastel1_5, #自定义pastel1
            'Pastel2_5': pastel2_5, #自定义pastel2（5种颜色）
            'Pastel2_10': pastel2_10,  # 自定义pastel2（10种颜色）
            'Pastel_coolwarm': create_pastel_coolwarm()
        }
        return cmap_options.get(cmap_name, plt.cm.Pastel1)

    # 自动选择文本颜色的函数
    def get_text_color(cell_value, cmap, vmin=0, vmax=1):
        """根据单元格背景颜色自动选择文本颜色"""
        if text_color == 'auto':
            # 根据单元格值计算颜色亮度
            norm_value = (cell_value - vmin) / (vmax - vmin)
            rgba = cmap(norm_value)
            # 计算亮度 (YIQ颜色空间)
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            return 'black' if brightness < 0.5 else 'black'
        else:
            return text_color

    # 获取颜色映射
    selected_cmap = get_cmap(cmap)

    # 创建热力图
    fig, ax = plt.subplots(figsize=figsize)

    # 判断是否使用离散色彩映射（10种颜色）
    # 如果是自定义的10种颜色映射，我们使用离散的颜色边界
    if cmap in ['Pastel2_10']:
        # 创建离散颜色边界：10个区间，每个区间0.1
        from matplotlib.colors import BoundaryNorm

        # 定义边界：从0到1，步长为0.1，总共11个边界值
        bounds = np.arange(0, 1.1, 0.1)

        # n_bins = 10
        # bounds = np.linspace(vmin, vmax, n_bins + 1)

        # 创建BoundaryNorm，将数据映射到离散的颜色区间
        norm = BoundaryNorm(bounds, selected_cmap.N)

        # 对于分开显示，我们需要处理NaN值
        if row_labels_type == 'separate':
            mask = np.isnan(data_matrix)
            heatmap = sns.heatmap(data_matrix,
                                  annot=show_values,
                                  fmt='.2f' if show_values else '',
                                  cmap=selected_cmap,
                                  norm=norm,  # 使用离散的norm
                                  linewidths=0.5,
                                  linecolor='gray',
                                  mask=mask,
                                  cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8,
                                            'ticks': bounds[:-1] + 0.05,  # 设置刻度在区间中间
                                            'boundaries': bounds},  # 设置边界
                                  annot_kws={'size': 12, 'color': text_color} if show_values else None,
                                  ax=ax)
        else:
            heatmap = sns.heatmap(data_matrix,
                                  annot=show_values,
                                  fmt='.2f' if show_values else '',
                                  cmap=selected_cmap,
                                  norm=norm,  # 使用离散的norm
                                  linewidths=0.5,
                                  linecolor='gray',
                                  cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8,
                                            'ticks': bounds[:-1] + 0.05,  # 设置刻度在区间中间
                                            'boundaries': bounds},  # 设置边界
                                  annot_kws={'size': 12, 'color': text_color} if show_values else None,
                                  ax=ax)
    else:
        # 对于分开显示，我们需要处理NaN值
        if row_labels_type == 'separate':
            mask = np.isnan(data_matrix)
            heatmap = sns.heatmap(data_matrix,
                                  annot=show_values,
                                  fmt='.2f' if show_values else '',
                                  cmap=selected_cmap,
                                  center=0,
                                  # vmin=0,
                                  # vmax=1,
                                  vmin=vmin,
                                  vmax=vmax,
                                  linewidths=0.5,
                                  linecolor='gray',
                                  mask=mask,
                                  cbar_kws={'label': 'Paired T-Test Statistic', 'shrink': 0.8},
                                  annot_kws={'size': 12, 'color': text_color} if show_values else None,
                                  ax=ax)
        else:
            heatmap = sns.heatmap(data_matrix,
                                  annot=show_values,
                                  fmt='.2f' if show_values else '',
                                  cmap=selected_cmap,
                                  center=0,
                                  # vmin=0,
                                  # vmax=1,
                                  vmin=vmin,
                                  vmax=vmax,
                                  linewidths=0.5,
                                  linecolor='gray',
                                  cbar_kws={'label': 'Paired T-Test Statistic', 'shrink': 0.8},
                                  annot_kws={'size': 12, 'color': text_color} if show_values else None,
                                  ax=ax)

    # 如果使用自动文本颜色，需要单独设置每个单元格的文本颜色
    if show_values and text_color == 'auto':
        # 遍历所有文本框，根据背景色设置文本颜色
        for i, text_row in enumerate(heatmap.texts):
            # 计算文本对应的矩阵位置
            n_rows, n_cols = data_matrix.shape
            # 热力图文本是按行顺序排列的
            row = i // n_cols
            col = i % n_cols
            # 跳过被掩码的单元格
            if row_labels_type == 'separate' and np.isnan(data_matrix[row, col]):
                continue

            cell_value = data_matrix[row, col]
            # 获取文本颜色
            txt_color = get_text_color(cell_value, selected_cmap)
            text_row.set_color(txt_color)
            text_row.set_fontweight('bold')

    # 自定义标签
    ax.set_xlabel(f"{metric}", fontsize=12)# fontweight='bold'
    ax.set_ylabel('Prompt Types', fontsize=12)# fontweight='bold'

    # 设置x轴标签（模型）
    ax.set_xticks(np.arange(len(models)) + 0.5)
    ax.set_xticklabels(models_name, fontsize=12, rotation=0)# fontweight='bold',

    # 设置y轴标签（测试类型 + 语言）
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    yticklabels = []
    for label in row_labels:
        # 为分隔行添加粗体
        if row_labels_type == 'separate' and label.endswith(')'):
            yticklabels.append(label)
        else:
            yticklabels.append(label)
    ax.set_yticklabels(yticklabels, fontsize=11, rotation=360)#, fontweight='bold'

    # 添加显著性标记（如果需要）
    if include_pvalue and show_values:
        for i in range(data_matrix.shape[0]):
            for j in range(data_matrix.shape[1]):
                # 跳过NaN值（对于分开显示模式）
                if row_labels_type == 'separate' and np.isnan(data_matrix[i, j]):
                    continue

                p_value = pvalue_matrix[i][j]
                if isinstance(p_value, (int, float)) and p_value < pvalue_threshold:
                    # 根据显著性水平添加不同数量的星号
                    if p_value < 0.001:
                        star = '***'
                    elif p_value < 0.01:
                        star = '**'
                    elif p_value < pvalue_threshold:
                        star = '*'
                    else:
                        star = ''

                    if star:
                        # 在单元格中添加星号
                        ax.text(j + 0.5, i + 0.9, star,
                                ha='center', va='center',
                                fontsize=text_size,
                                color='dimgray') #fontweight='bold'

    # # 设置标题
    # ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.set_ticklabels([f"{x:.2f}" for x in np.linspace(vmin, vmax, 5)])
    cbar.ax.tick_params(labelsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存至: {save_path}")

    # 显示图表
    plt.show()

    return fig, ax

# 绘制Spearman相关系数散点图
def plot_Spearman_scatter(spearman_data_dict, title="Model Performance Comparison (Spearman Correlation)", figsize=(12, 6), show_labels=True, save_path=None, repeat=True):

    # 提取模型名称
    models = list(spearman_data_dict.keys())

    if len(models) == 2:
        models_name = ['Qwen7B', 'CantoLLM7B']
    elif len(models) == 5:
        models_name = ['Qwen', 'DeepSeek', 'GPT', 'Mistral', 'Llama']
    else:
        # 默认情况：使用models中的名称，首字母大写
        models_name = [model.replace('_', ' ').title() for model in models]
        print(f"警告: 检测到 {len(models)} 个模型，使用默认名称: {models_name}")

    # 定义语言和测试类型
    languages = ['en', 'zh', 'ca']
    test_types = ['zero_shot', 'few_shot']

    # 准备Spearman数据数组
    spearman_data_arrays = {}

    for test_type in test_types:
        for lang in languages:
            key = f"{test_type}_{lang}"
            if repeat:
                spearman_data_arrays[key] = [spearman_data_dict[model]['duplicate'][test_type][lang]['spearman'] for
                                             model in models]
            else:
                spearman_data_arrays[key] = [spearman_data_dict[model]['deduplicate'][test_type][lang]['spearman'] for
                                             model in models]

    # 设置图表
    fig, ax = plt.subplots(figsize=figsize)

    # 设置x轴位置
    x = np.arange(len(models))
    width = 0.14  # 保持与柱状图相同的宽度概念 #0.14

    # 为Spearman点定义形状和颜色
    spearman_markers = {
        'zero_shot_en': {'marker': 'o', 'color': 'none', 'edgecolor': '#E69F00', 'size': 40},
        'zero_shot_zh': {'marker': 'o', 'color': 'none', 'edgecolor': '#7A9E3A', 'size': 40},
        'zero_shot_ca': {'marker': 'o', 'color': 'none', 'edgecolor': '#3B6FA5', 'size': 40},
        'few_shot_en': {'marker': 'o', 'color': '#E69F00', 'edgecolor': 'none', 'size': 40},
        'few_shot_zh': {'marker': 'o', 'color': '#7A9E3A', 'edgecolor': 'none', 'size': 40},
        'few_shot_ca': {'marker': 'o', 'color': '#3B6FA5', 'edgecolor': 'none', 'size': 40}
    }

    # 绘制Spearman散点
    scatters = []
    for i, (test_type, lang) in enumerate([(tt, lang) for tt in test_types for lang in languages]):
        key = f"{test_type}_{lang}"
        offset = (i - 2.5) * width  # 调整位置

        # 计算实际x位置
        x_pos = x + offset

        # 获取Spearman数据和标记设置
        spearman_values = spearman_data_arrays[key]
        marker_style = spearman_markers[key]

        # 绘制浅灰色垂直线（从x轴到每个散点）
        for j in range(len(x_pos)):
            # 绘制垂直线
            ax.plot([x_pos[j], x_pos[j]], [0, spearman_values[j]],
                   color='lightgray',
                   linewidth=0.8,
                   alpha=0.5,
                   zorder=1)  # 低zorder确保在散点下方

        # 绘制散点
        scatter = ax.scatter(x_pos, spearman_values,
                             s=marker_style['size'],
                             marker=marker_style['marker'],
                             color=marker_style['color'],
                             edgecolor=marker_style['edgecolor'],
                             linewidth=1.5,
                             zorder=5,
                             label=f"{test_type.replace('_', '-').title()} ({lang})")
        scatters.append(scatter)

    # 设置y轴
    ax.set_ylabel('Spearman Correlation', fontsize=15, color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_ylim(0, 1.0)  # Spearman范围[0,1]

    # 设置标题和x轴标签
    ax.set_xlabel(metric, fontsize=15)
    # ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_name)

    ax.tick_params(axis='both', labelsize=14)  # 同时设置x和y轴刻度标签大小

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.axvline(x=0.5, color='gray', linewidth=1, linestyle='--', zorder=0, alpha=0.6)
    # ax.axvline(x=1.5, color='gray', linewidth=1, linestyle='--', zorder=0, alpha=0.6)
    # ax.axvline(x=2.5, color='gray', linewidth=1, linestyle='--', zorder=0, alpha=0.6)
    # ax.axvline(x=3.5, color='gray', linewidth=1, linestyle='--', zorder=0, alpha=0.6)

    # 创建图例
    legend_elements = []
    for i, (test_type, lang) in enumerate([(tt, lang) for tt in test_types for lang in languages]):
        key = f"{test_type}_{lang}"
        marker_style = spearman_markers[key]

        # 创建散点标记
        line = Line2D([0], [0], marker=marker_style['marker'], color='w',
                      markerfacecolor=marker_style['color'],
                      markeredgecolor=marker_style['edgecolor'],
                      markersize=10,
                      label=f"{test_type.replace('_', '-').title()} ({lang})")
        legend_elements.append(line)

    # # 添加图例
    # ax.legend(handles=legend_elements,
    #           loc='lower center',
    #           # bbox_to_anchor=(0.5, -0.15),
    #           ncol=6,
    #           fontsize=14,
    #           frameon=True,
    #           framealpha=0.9)

    # 调整布局，为底部图例留出空间
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # 添加数据标签
    if show_labels:
        # 获取y轴的最大值，用于计算合适的偏移量
        y_max = ax.get_ylim()[1]
        offset = y_max * 0.03

        for i, (test_type, lang) in enumerate([(tt, lang) for tt in test_types for lang in languages]):
            key = f"{test_type}_{lang}"
            marker_offset = (i - 2.5) * width
            x_pos = x + marker_offset

            for j, value in enumerate(spearman_data_arrays[key]):
                # 位置在散点上方
                ax.text(x_pos[j], value + offset,
                        f'{value:.2f}', ha='center', va='bottom',
                        fontsize=9, color='black') #fontweight='bold'

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    # 显示图表
    plt.show()

    return fig, ax

# 可视化Wilcoxon检验结果 - 配对连线图
def visualize_wilcoxon_results(same, different, results, save_path):
    """
    可视化Wilcoxon检验结果 - 配对连线图

    参数:
        same: same组的MAE值列表
        different: different组的MAE值列表
        results: Wilcoxon检验结果字典
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(4, 4.5))

    # 绘制配对连线
    for i in range(len(same)):
        ax.plot([1, 2], [same[i], different[i]], 'o-',
                alpha=0.6, linewidth=0.8, markersize=6, color='dimgray')

    ax.set_ylim(0, 4)  # Spearman范围[0,1]

    # 计算均值和中位数
    same_mean = np.mean(same)
    different_mean = np.mean(different)
    same_median = np.median(same)
    different_median = np.median(different)

    # 绘制均值线
    ax.plot([1, 2], [same_mean, different_mean], 'b-o',
            linewidth=3, markersize=11, color='RoyalBlue', label='Mean')

    # 绘制中位数线
    ax.plot([1, 2], [same_median, different_median], 'b-o',
            linewidth=3, markersize=11, color='SeaGreen', label='Median')

    # 在均值点旁边添加数值标签
    # Same组均值
    ax.text(0.95, same_mean, f'{same_mean:.2f}',
            color='RoyalBlue', fontsize=14, ha='right', va='center',
            )#bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

    # Different组均值
    ax.text(2.05, different_mean, f'{different_mean:.2f}',
            color='RoyalBlue', fontsize=14, ha='left', va='center',
            )#bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

    # 在中位数点旁边添加数值标签
    # Same组中位数
    ax.text(0.95, same_median, f'{same_median:.2f}',
            color='SeaGreen', fontsize=14, ha='right', va='center',
            )#bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

    # Different组中位数
    ax.text(2.05, different_median, f'{different_median:.2f}',
            color='SeaGreen', fontsize=14, ha='left', va='center',
            )#bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

    # 在图片左上角添加统计信息
    if results and 'z_score' in results and 'p_value' in results and 'effect_size_r' in results:
        z_val = results['z_score']
        p_val = results['p_value']
        r_val = results['effect_size_r']

        # 将p值格式化为适当的形式
        if p_val < 0.001:
            p_str = "<0.001"
        elif p_val < 0.01:
            p_str = f"={p_val:.2f}"
        else:
            p_str = f"={p_val:.2f}"

        # 确定显著性标记
        if p_val < 0.001:
            sig_marker = "***"
        elif p_val < 0.01:
            sig_marker = "**"
        elif p_val < 0.05:
            sig_marker = "*"
        else:
            sig_marker = ""

        # 确定效力标记
        if r_val < 0.1:
            sig_marker1 = ""
        elif r_val < 0.3:
            sig_marker1 = "*"
        elif r_val < 0.5:
            sig_marker1 = "**"
        else:
            sig_marker1 = "***"

        # 添加统计信息文本
        stats_text = f"z: {z_val:.2f} {sig_marker}\nr: {r_val:.2f} {sig_marker1}" #{p_val:.2f}

        # 添加文本框，位置在左上角，使用相对坐标
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=13, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='lightgray'))

    # 添加图例（替代原来的文本标注）
    ax.legend(loc='upper center', bbox_to_anchor=(1.05, 1), ncol=2, frameon=True, framealpha=0.9)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置x轴
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Same', 'Different'], fontsize=14)
    ax.set_xlabel('Familiarity', fontsize=14)

    # 设置y轴
    ax.set_ylabel('Mean Absolute Error', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    # 调整布局
    plt.tight_layout()
    plt.show()

    return fig, ax

# 可视化Wilcoxon检验结果 - 配对连线图 (for DeepSeek + GPT)
def visualize_DepSeek_GPT_MAE_results(same, different, save_path):

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # 绘制配对连线
    for i in range(len(same)):
        ax.plot([1, 2], [same[i], different[i]], 'o-',
                alpha=0.6, linewidth=0.8, markersize=6, color='dimgray')

    ax.set_ylim(0, 3.5)

    # 计算均值和中位数
    same_mean = np.mean(same)
    different_mean = np.mean(different)
    same_median = np.median(same)
    different_median = np.median(different)

    # 绘制均值线
    ax.plot([1, 2], [same_mean, different_mean], 'b-o',
            linewidth=3, markersize=11, color='RoyalBlue', label='Mean')

    # 绘制中位数线
    ax.plot([1, 2], [same_median, different_median], 'b-o',
            linewidth=3, markersize=11, color='SeaGreen', label='Median')

    # 在均值点旁边添加数值标签
    # Same组均值
    ax.text(0.95, same_mean, f'{same_mean:.2f}',
            color='RoyalBlue', fontsize=14, ha='right', va='center',
            )#bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

    # Different组均值
    ax.text(2.05, different_mean, f'{different_mean:.2f}',
            color='RoyalBlue', fontsize=14, ha='left', va='center',
            )#bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

    # 在中位数点旁边添加数值标签
    # Same组中位数
    ax.text(0.95, same_median, f'{same_median:.2f}',
            color='SeaGreen', fontsize=14, ha='right', va='center',
            )#bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

    # Different组中位数
    ax.text(2.05, different_median, f'{different_median:.2f}',
            color='SeaGreen', fontsize=14, ha='left', va='center',
            )#bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置x轴
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Same', 'Different'], fontsize=14)
    ax.set_xlabel('AoA', fontsize=14)

    # 设置y轴
    ax.set_ylabel('Mean Absolute Error', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    # 调整布局
    plt.tight_layout()
    plt.show()

    return fig, ax

# 绘制model RDM图
def create_pastel_coolwarm():
    """创建柔和的coolwarm配色"""
    colors = [
        # (1.0, 1.0, 1.0, 1.0), #纯白色
        (0.96, 0.96, 0.96, 1.0), # 几乎白色

        # (0.95, 0.92, 0.9, 1.0),  # 淡红色
        # (0.95, 0.8, 0.7, 1.0),   # 中等红色 - 比原版稍深
        # (0.85, 0.6, 0.4, 1.0)    # 加深的红色 - 更饱和的红色
        (0.9, 0.92, 0.95, 1.0),  # 淡蓝色
        (0.7, 0.8, 0.95, 1.0),   # 中等蓝色 - 比原版稍深
        (0.4, 0.6, 0.85, 1.0)   # 加深的蓝色 - 更饱和的蓝色

    ]
    return LinearSegmentedColormap.from_list('pastel_coolwarm', colors, N=256)

def visualize_rdms(model_names, correlation_matrices, savepath, figsize=(20, 10)):
    """
    可视化12个相关矩阵的热力图

    参数:
        model_names: 包含12个模型名称的列表
        correlation_matrices: 包含12个相关矩阵的列表，每个矩阵为4x4
        figsize: 图形尺寸
    """
    # 检查输入参数
    if len(model_names) != 12 or len(correlation_matrices) != 12:
        raise ValueError("需要12个模型名称和12个相关矩阵")

    # 创建图形，2行6列
    fig, axes = plt.subplots(2, 6, figsize=figsize)
    # fig.suptitle('Variable Correlation Matrices Across Different Models', fontsize=16, y=1.02)

    # 定义标签
    # labels = ['AoA', 'Familiarity', 'Concreteness', 'Imageability']
    labels = ['AoA', 'FAM', 'CON', 'IMA']

    # 为每个子图绘制热力图
    for i in range(12):
        # 计算行和列索引
        row = i // 6
        col = i % 6

        # 获取当前轴
        ax = axes[row, col]

        # 获取当前模型的相关矩阵
        corr_matrix = correlation_matrices[i]

        # 确保矩阵是对称的
        # 创建完整的相关矩阵（如果只提供了上三角部分）
        full_matrix = np.zeros((4, 4))

        mask = np.zeros((4, 4), dtype=bool)

        for r in range(4):
            for c in range(4):
                if r == c:
                    full_matrix[r, c] = 0.0
                elif r < c and corr_matrix.shape[0] == 4:
                    # 如果已经是完整矩阵
                    full_matrix[r, c] = corr_matrix[r, c]
                else:
                    # 如果只提供了上三角部分，需要构建完整矩阵
                    # full_matrix[r, c] = corr_matrix[max(r, c), min(r, c)]
                    full_matrix[r, c] = -1
                    mask[r, c] = True  # 标记为掩码

        # # 创建掩码数组：将值为-1的位置标记为掩码
        # masked_array = np.ma.array(full_matrix, mask=(full_matrix == -1))
        # 创建掩码数组
        masked_array = np.ma.array(full_matrix, mask=mask)

        # 创建色彩映射并设置掩码值的颜色为白色
        cmap = create_pastel_coolwarm()

        # 绘制热力图
        im = ax.imshow(full_matrix, cmap=create_pastel_coolwarm(), vmin=0, vmax=2)

        # 设置标题
        ax.set_title(f'{model_names[i]}', fontsize=17, pad=15)

        # 设置刻度
        if i >= 6:
          ax.set_xticks(range(4))
          ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=18)
          ax.tick_params(axis='x', pad=10)
        else:
          ax.set_xticks([])
        if i == 0 or i == 6:
          ax.set_yticks(range(4))
          ax.set_yticklabels(labels, rotation=90, va='center', fontsize=18)
          ax.tick_params(axis='y', pad=3)
        else:
          ax.set_yticks([])

        # 添加网格线
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # 在单元格中显示数值
        for r in range(4):
            for c in range(4):
              if r <= c:  # 只在上三角和对角线显示数值
                value = full_matrix[r, c]
                if not mask[r, c]:  # 不是掩码部分
                  # 根据背景颜色选择文本颜色
                  text_color = 'black' if abs(value) > 0.5 else 'black'
                  ax.text(c, r, f'{value:.2f}', ha='center', va='center',
                        color=text_color, fontsize=18)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 左边、底边、右边、顶边，这里右边留出10%的空间
    cbar_ax = fig.add_axes([0.9, 0.22, 0.01, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)

    # 保存图表
    if savepath:
      plt.savefig(savepath, dpi=300, bbox_inches='tight')
      print(f"图表已保存至: {savepath}")

    plt.show()
    return fig


# 绘制human RDM图
def create_pastel_coolwarm():
    """创建柔和的coolwarm配色"""
    colors = [
        # (1.0, 1.0, 1.0, 1.0), #纯白色
        (0.96, 0.96, 0.96, 1.0), # 几乎白色

        (0.95, 0.92, 0.9, 1.0),  # 淡红色
        (0.95, 0.8, 0.7, 1.0),   # 中等红色 - 比原版稍深
        (0.85, 0.6, 0.4, 1.0)    # 加深的红色 - 更饱和的红色
        # (0.9, 0.92, 0.95, 1.0),  # 淡蓝色
        # (0.7, 0.8, 0.95, 1.0),   # 中等蓝色 - 比原版稍深
        # (0.4, 0.6, 0.85, 1.0)   # 加深的蓝色 - 更饱和的蓝色
    ]
    return LinearSegmentedColormap.from_list('pastel_coolwarm', colors, N=256)

def visualize_single_rdm(corr_matrix, model_name, savepath=None, figsize=(8, 6)):
    """
    可视化单个相关矩阵的热力图

    参数:
        corr_matrix: 4x4相关矩阵（可以是上三角或完整矩阵）
        model_name: 模型名称（用于标题）
        savepath: 保存路径，如果为None则不保存
        figsize: 图形尺寸
    """
    # 创建图形，只有一个子图
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 定义标签
    # labels = ['AoA', 'Familiarity', 'Concreteness', 'Imageability']
    labels = ['AoA', 'FAM', 'CON', 'IMA']

    # 确保矩阵是对称的
    # 创建完整的相关矩阵（如果只提供了上三角部分）
    full_matrix = np.zeros((4, 4))
    mask = np.zeros((4, 4), dtype=bool)

    for r in range(4):
        for c in range(4):
            if r == c:
                full_matrix[r, c] = 0.0
            elif r < c and corr_matrix.shape[0] == 4:
                # 如果已经是完整矩阵
                full_matrix[r, c] = corr_matrix[r, c]
                full_matrix[c, r] = corr_matrix[r, c]  # 确保对称
            else:
                # # 如果只提供了上三角部分，需要构建完整矩阵
                # full_matrix[r, c] = corr_matrix[max(r, c), min(r, c)]
                full_matrix[r, c] = -1
                mask[r, c] = True  # 标记为掩码

    # 创建掩码数组
    masked_array = np.ma.array(full_matrix, mask=mask)

    # 绘制热力图
    im = ax.imshow(full_matrix, cmap=create_pastel_coolwarm(), vmin=0, vmax=2)

    # 设置标题
    ax.set_title(f'{model_name}', fontsize=22, pad=20)

    # 设置刻度
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=20)
    ax.set_yticklabels(labels, rotation=90, va='center', fontsize=20)
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=3)

    # 添加网格线
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 在单元格中显示数值
    for r in range(4):
        for c in range(4):
          if r <= c:  # 只在上三角和对角线显示数值
            value = full_matrix[r, c]
            if not mask[r, c]:  # 不是掩码部分
              # 根据背景颜色选择文本颜色
              text_color = 'black' if abs(value) > 0.5 else 'black'
              ax.text(c, r, f'{value:.2f}', ha='center', va='center',
                    color=text_color, fontsize=24)

    # 调整布局，添加颜色条
    plt.tight_layout(rect=[0, 0, 0.97, 1])  # 右边留出5%的空间给颜色条

    # 添加颜色条
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    # cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('Correlation Coefficient', fontsize=14)

    # 保存图表
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {savepath}")

    plt.show()
    return fig

if __name__ == "__main__":
    pass





