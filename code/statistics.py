# -*- coding:utf-8 -*-
'''
@file   :statistics.py
@author :Floret Huacheng SONG
@time   :25/11/2025 下午6:19
@purpose: for computing statistical test
'''

from sklearn.metrics import mean_absolute_error
from scipy import stats
import numpy as np
import json
from typing import Any, Dict, List, Optional
from model import data_generation
import pingouin as pg
import os
import pandas as pd
import matplotlib.pyplot as plt
# import rsatoolbox as rsa
from scipy.stats import spearmanr
from visulization import read_MAEexcel_to_dict
from scipy.stats import norm

from scipy.stats import wilcoxon, rankdata
from visulization import visualize_wilcoxon_results

from scipy.stats import mannwhitneyu

class MergedDataParser:

    def __init__(self, inpath: str):
        self.inpath = inpath
        self.data = None
        self._load_data()

    def _load_data(self) -> None:
        try:
            with open(self.inpath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print("数据加载成功！")
        except Exception as e:
            print(f"读取文件时出错: {e}")
            self.data = None

    def get_characters(self, language: str) -> List[str]:
        if not self.data or language not in self.data:
            return []
        return list(self.data[language].keys())

    def get_character_info(self, language: str, character: str) -> Optional[Dict]:
        if not self.data:
            return None

        return self.data.get(language, {}).get(character)

    def get_character_value(self, language: str, character: str) -> Optional[str]:
        info = self.get_character_info(language, character)
        if info:
            return info.get('character', character)
        return None

    def get_id_value(self, language: str, character: str) -> Optional[str]:
        info = self.get_character_info(language, character)
        if info:
            return info.get('id')
        return None

    def get_num_value(self, language: str, character: str) -> Optional[str]:
        info = self.get_character_info(language, character)
        if info:
            return info.get('num', 1)
        else:
            return 1

    def get_simplified_value(self, language: str, character: str) -> Optional[str]:
        info = self.get_character_info(language, character)
        if info:
            return info.get('simple_character', character)
        return None

    def get_all_results(self, language: str, character: str) -> Dict[str, Any]:
        info = self.get_character_info(language, character)
        if info:
            return info.get('results', {})
        return {}

    def get_tasks(self, language: str, character: str) -> List[str]:
        info = self.get_character_info(language, character)
        if info and 'results' in info:
            return list(info['results'].keys())
        return []

    def get_modes(self, language: str, character: str, task: str) -> List[str]:
        info = self.get_character_info(language, character)
        if info and 'results' in info and task in info['results']:
            return list(info['results'][task].keys())
        return []

    def get_types(self, language: str, character: str, task: str, mode: str) -> List[str]:
        info = self.get_character_info(language, character)
        if (info and 'results' in info and
                task in info['results'] and
                mode in info['results'][task]):
            return list(info['results'][task][mode].keys())
        return []

    def get_models(self, language: str, character: str, task: str, mode: str, type_: str) -> List[str]:
        info = self.get_character_info(language, character)
        if (info and 'results' in info and
                task in info['results'] and
                mode in info['results'][task] and
                type_ in info['results'][task][mode]):
            return list(info['results'][task][mode][type_].keys())
        return []

    def get_values(self, language: str, character: str, task: str, mode: str,
                   type_: str, model: str) -> Optional[List[Any]]:
        info = self.get_character_info(language, character)
        if (info and 'results' in info and
                task in info['results'] and
                mode in info['results'][task] and
                type_ in info['results'][task][mode] and
                model in info['results'][task][mode][type_]):
            return info['results'][task][mode][type_][model]
        return None

def compute_MAE(true: list, pred: list):
    # 真实值
    y_true = np.array(true)
    # 预测值
    y_pred = np.array(pred)

    # 计算MAE
    mae = mean_absolute_error(y_true, y_pred)
    # print(f"MAE: {mae}")  # 输出： MAE: 11.0
    return mae, 1, 1

def compute_PAIREDT(true: list, pred: list):
    # 原始数据
    y_true = np.array(true)
    y_pred = np.array(pred)

    t_stat, p_value = stats.ttest_rel(y_true, y_pred)

    if p_value < 0.05:
        print("差异在统计上是显著的。")
        return True, t_stat, p_value
    else:
        print("差异在统计上不显著。")
        return False, t_stat, p_value

def compute_SPEARMAN(true: list, pred: list):
    # 原始数据
    y_true = np.array(true)
    y_pred = np.array(pred)

    correlation, p_value = stats.spearmanr(y_true, y_pred)

    # print(f"斯皮尔曼相关系数 (ρ): {correlation:.4f}")
    # print(f"P值: {p_value:.4f}")

    if p_value < 0.05:
        print("相关性在统计上是显著的。")
        return True, correlation, p_value
    else:
        print("相关性在统计上不显著。")
        return False, correlation, p_value

def compute_KENDALL(true: list, pred: list):
    # 原始数据
    y_true = np.array(true)
    y_pred = np.array(pred)

    # 计算Kendall's Tau（三种形式）
    tau_b, p_value = stats.kendalltau(y_true, y_pred)

    if p_value < 0.05:
        print("相关性在统计上是显著的。")
        return True, tau_b, p_value
    else:
        print("相关性在统计上不显著。")
        return False, tau_b, p_value

def compute_ICC(len, true: list, pred: list):
    # 原始数据
    y_true = np.array(true)
    y_pred = np.array(pred)


    df = pd.DataFrame({
        'rater': ['human'] * len + ['machine'] * len,
        'score': list(y_true) + list(y_pred),
        'item': list(range(len)) * 2  # 每个样本的ID
    })

    # 计算ICC - 建议使用ICC(2,1)或ICC(3,1)
    icc_result = pg.intraclass_corr(
        data=df,
        targets='item',
        raters='rater',
        ratings='score'
    )

    print(icc_result)
    return icc_result, 1, 1

def bland_altman_analysis(data1, data2, title="Bland-Altman图"):
    """
    Bland-Altman一致性分析
    """
    # 确保数据是NumPy数组
    data1 = np.array(data1)
    data2 = np.array(data2)

    # 确保两个数组长度相同
    if len(data1) != len(data2):
        raise ValueError(f"数据长度不一致: data1={len(data1)}, data2={len(data2)}")

    # 计算均值和差值
    means = np.mean([data1, data2], axis=0)
    diffs = data1 - data2

    # 计算一致性界限
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    lower_loa = mean_diff - 1.96 * std_diff
    upper_loa = mean_diff + 1.96 * std_diff

    # 创建图形
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 散点图
    ax[0].scatter(data1, data2, alpha=0.5)
    ax[0].plot([data1.min(), data1.max()],
               [data1.min(), data1.max()], 'r--', label='y=x')
    ax[0].set_xlabel('人工评分')
    ax[0].set_ylabel('机器评分')
    ax[0].set_title(f'{title} - 散点图')
    ax[0].legend()

    # Bland-Altman图
    ax[1].scatter(means, diffs, alpha=0.5)
    ax[1].axhline(mean_diff, color='red', linestyle='--', label=f'均值差: {mean_diff:.3f}')
    ax[1].axhline(lower_loa, color='gray', linestyle='--', label=f'LoA下限: {lower_loa:.3f}')
    ax[1].axhline(upper_loa, color='gray', linestyle='--', label=f'LoA上限: {upper_loa:.3f}')
    ax[1].set_xlabel('平均评分')
    ax[1].set_ylabel('差值(人工-机器)')
    ax[1].set_title(f'{title} - Bland-Altman图')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 返回统计结果
    return {
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'lower_loa': lower_loa,
        'upper_loa': upper_loa,
        'proportion_within_loa': np.sum((diffs >= lower_loa) & (diffs <= upper_loa)) / len(diffs)
    }, 1, 1

def compute_results(language, type_, task, mode, model, metric, repeat):

    input_path = r'@merged.xlsx'
    input_dict = data_generation(input_path, language)
    # print(input_dict)
    character_list = list(input_dict.keys())
    # print(character_list)

    results_path = r"merged_rounds_updated.json"
    parser = MergedDataParser(results_path)

    gold_results = []
    results = []

    for kt, vt in input_dict.items():
        number = parser.get_num_value(language, kt)
        if not repeat:
            number = 1
        # print(number)
        if task == "AoA":
            for _ in range(number):
                gold_results.append(vt['gold_AoA'])
        elif task == "familiarity":
            for _ in range(number):
                gold_results.append(vt['gold_Familiarity'])
        elif task == "concreteness":
            for _ in range(number):
                gold_results.append(vt['gold_Concreteness'])
        elif task == "imageability":
            for _ in range(number):
                gold_results.append(vt['gold_Imageability'])
        else:
            print(f"Wrong {task}!")
    # print(len(gold_results))

    for c in character_list:
        # id = parser.get_id_value(language, c)
        number = parser.get_num_value(language, c)
        if not repeat:
            number = 1
        # print(number)
        score_list = parser.get_values(language, c.strip('1'), task, mode, type_, model)
        avg_score = score_list[3]
        for _ in range(number):
            results.append(avg_score)
    print(results)

    if metric == "mae":
        state, correlation, p_value = compute_MAE(gold_results, results)
    elif metric == "pairedt":
        state, correlation, p_value = compute_PAIREDT(gold_results, results)
    elif metric == "spearman":
        state, correlation, p_value = compute_SPEARMAN(gold_results, results)
    elif metric == "kendall":
        state, correlation, p_value = compute_KENDALL(gold_results, results)
    elif metric == "icc":
        state, correlation, p_value = compute_ICC(len(results), gold_results, results)
    elif metric == "ba":
        state, correlation, p_value = bland_altman_analysis(gold_results, results)
    else:
        print(f"Wrong {metric}!")

    return state, correlation, p_value

class RDM_analysis:

    def __init__(self, inpath, language, mode, type_, model, repeat=True):
        self.inpath = inpath
        self.language = language
        self.type = type_
        # self.task = task
        self.mode = mode
        self.model = model
        self.repeat = repeat
        self.datasets = []

        input_path = r'@merged.xlsx'
        input_dict = data_generation(input_path, language)
        self.character_list = list(input_dict.keys())
        # print(character_list)

        self.parser = MergedDataParser(self.inpath)

        self.gold_results_AoA = []
        self.gold_results_familiarity = []
        self.gold_results_concreteness = []
        self.gold_results_imageability = []

        for kt, vt in input_dict.items():
            number = self.parser.get_num_value(language, kt)
            if not repeat:
                number = 1
            for _ in range(number):
                self.gold_results_AoA.append(vt['gold_AoA'])
                self.gold_results_familiarity.append(vt['gold_Familarity'])
                self.gold_results_concreteness.append(vt['gold_Concreteness'])
                self.gold_results_imageability.append(vt['gold_Imageability'])

    def load_model_data(self, task):
        self.score_list = []

        for c in self.character_list:
            number = self.parser.get_num_value(self.language, c)

            if not self.repeat:
                number = 1

            scores = self.parser.get_values(self.language, c.strip('1'), task, self.mode, self.type, self.model)
            avg_score = scores[3]
            for _ in range(number):
                self.score_list.append(avg_score)

        print(len(self.score_list))
        self.data = np.array(self.score_list)
        return self.data

    # 计算4个变量之间的RDM（距离 = 1 - 相关系数）
    def compute_variable_correlation_rdm(self, data_type='model', correlation='pearson'):
        if data_type == 'model':
            # 获取模型数据
            data_df = pd.DataFrame({
                'AoA': np.array(self.load_model_data('AoA')),
                'familiarity': np.array(self.load_model_data('familiarity')),
                'concreteness': np.array(self.load_model_data('concreteness')),
                'imageability': np.array(self.load_model_data('imageability'))
            })
            measurements = data_df.T.values  # 形状: (4, n_items)
        elif data_type == 'gold':
            # 使用人类金标准数据（已经在__init__中计算好了）
            measurements = np.vstack([
                self.gold_results_AoA,
                self.gold_results_familiarity,
                self.gold_results_concreteness,
                self.gold_results_imageability
            ])  # 形状: (4, n_items)
        else:
            raise ValueError("data_type 必须是 'model' 或 'gold'")

        # 计算相关性矩阵
        if correlation == 'pearson':
            corr_matrix = np.corrcoef(measurements)
        elif correlation == 'spearman':
            # 使用pandas计算Spearman相关
            df = pd.DataFrame(measurements.T, columns=['AoA', 'familiarity', 'concreteness', 'imageability'])
            corr_matrix = df.corr(method='spearman').values
            # spearman_corr, spearman_p = spearmanr(human_distances, model_distances)
        else:
            raise ValueError(f"不支持的相关性方法: {correlation}")

        # 转换为距离矩阵: 距离 = 1 - 相关系数
        dist_matrix = 1 - corr_matrix

        # 创建RDM对象
        # 提取上三角部分（不包括对角线）
        # n_vars = 4
        variable_names = ['AoA', 'familiarity', 'concreteness', 'imageability']
        triu_indices = np.triu_indices(4, k=1)
        dissimilarities = dist_matrix[triu_indices].reshape(1, -1)

        rdm = rsa.rdm.RDMs(
            dissimilarities=dissimilarities,
            dissimilarity_measure=f'1-{correlation}_correlation',
            descriptors={'data_type': data_type, 'model': self.model, 'language': self.language},
            rdm_descriptors={'data_type': [data_type], 'model': [self.model], 'language': [self.language]},
            pattern_descriptors={'variable': variable_names}
        )

        # 保存额外的矩阵信息
        rdm.correlation_matrix = corr_matrix
        rdm.distance_matrix = dist_matrix

        print(f"{data_type.upper()} 变量RDM计算完成:")
        print(f"相关性矩阵:\n{np.round(corr_matrix, 3)}")
        print(f"距离矩阵 (1-相关性):\n{np.round(dist_matrix, 3)}")

        return rdm, corr_matrix, dist_matrix

    # 比较人类和模型RDM（使用Spearman相关）
    def compare_human_model_rdms(self):
        # 计算人类金标准RDM
        human_rdm, human_corr, human_dist = self.compute_variable_correlation_rdm('gold', 'spearman')
        # 计算模型RDM
        model_rdm, model_corr, model_dist = self.compute_variable_correlation_rdm('model', 'spearman')

        # 方法2: 保持原来的Spearman相关作为参考
        # 提取距离向量（上三角部分）
        human_distances = human_rdm.dissimilarities[0]
        model_distances = model_rdm.dissimilarities[0]
        spearman_corr, spearman_p = spearmanr(human_distances, model_distances)

        return {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'human_rdm': human_rdm,
            'model_rdm': model_rdm,
            'human_correlation_matrix': human_corr,
            'model_correlation_matrix': model_corr,
            'human_distance_matrix': human_dist,
            'model_distance_matrix': model_dist,
        }

    def visualize_variable_rdms(self, results=None):
        """
        可视化人类和模型的变量RDM

        参数:
            results: 比较结果，如果不提供则重新计算
        """
        if results is None:
            results = self.compare_human_model_rdms()

        human_rdm = results['human_rdm']
        model_rdm = results['model_rdm']

        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Human vs {self.model} Variable Relationships', fontsize=16)

        # 1. 人类相关性矩阵
        ax1 = axes[0, 0]
        im1 = ax1.imshow(results['human_correlation_matrix'], cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_title('Human: Correlation Matrix')
        ax1.set_xticks(range(4))
        ax1.set_yticks(range(4))
        ax1.set_xticklabels(['AoA', 'Fam', 'Conc', 'Img'])
        ax1.set_yticklabels(['AoA', 'Fam', 'Conc', 'Img'])
        plt.colorbar(im1, ax=ax1)

        # 2. 模型相关性矩阵
        ax2 = axes[0, 1]
        im2 = ax2.imshow(results['model_correlation_matrix'], cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_title(f'{self.model}: Correlation Matrix')
        ax2.set_xticks(range(4))
        ax2.set_yticks(range(4))
        ax2.set_xticklabels(['AoA', 'Fam', 'Conc', 'Img'])
        ax2.set_yticklabels(['AoA', 'Fam', 'Conc', 'Img'])
        plt.colorbar(im2, ax=ax2)

        # 3. 人类RDM (1-相关性)
        ax3 = axes[0, 2]
        im3 = ax3.imshow(results['human_distance_matrix'], cmap='viridis')
        ax3.set_title('Human RDM (1 - correlation)')
        ax3.set_xticks(range(4))
        ax3.set_yticks(range(4))
        ax3.set_xticklabels(['AoA', 'Fam', 'Conc', 'Img'])
        ax3.set_yticklabels(['AoA', 'Fam', 'Conc', 'Img'])
        plt.colorbar(im3, ax=ax3)

        # 4. 模型RDM (1-相关性)
        ax4 = axes[1, 0]
        im4 = ax4.imshow(results['model_distance_matrix'], cmap='viridis')
        ax4.set_title(f'{self.model} RDM (1 - correlation)')
        ax4.set_xticks(range(4))
        ax4.set_yticks(range(4))
        ax4.set_xticklabels(['AoA', 'Fam', 'Conc', 'Img'])
        ax4.set_yticklabels(['AoA', 'Fam', 'Conc', 'Img'])
        plt.colorbar(im4, ax=ax4)

        # 5. 使用rsatoolbox显示人类RDM
        ax5 = axes[1, 1]
        rsa.vis.show_rdm(human_rdm, ax=ax5, pattern_descriptor='variable')
        ax5.set_title('Human RDM (rsatoolbox)')

        # 6. 使用rsatoolbox显示模型RDM
        ax6 = axes[1, 2]
        rsa.vis.show_rdm(model_rdm, ax=ax6, pattern_descriptor='variable')
        ax6.set_title(f'{self.model} RDM (rsatoolbox)')

        # 添加文本摘要
        fig.text(0.5, 0.01,
                 f"Spearman Correlation (Human vs {self.model}): {results['spearman_correlation']:.3f}, p = {results['p_value']:.4f}",
                 ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        return fig

    def run_full_analysis(self):
        """
        运行完整的分析流程
        """
        print("=" * 60)
        print(f"开始分析: {self.language} - {self.model} - {self.mode}-{self.type}")
        print("=" * 60)

        # 比较人类和模型RDM
        results = self.compare_human_model_rdms()

        # 可视化
        self.visualize_variable_rdms(results)

        return results

def Spearman_frequency_stroke(language, repeat):
    inpath = r"combined.xlsx"
    combined_pd = pd.read_excel(inpath)

    if language == 'simple':
        freq_list = combined_pd["S_Freq.log"].tolist()
        stroke_list = combined_pd["S_Stroke"].tolist()
        char_list = combined_pd["S_Character"].tolist()  # 新增：获取字符列表
    elif language == 'traditional':
        freq_list = combined_pd["T_Freq.log"].tolist()
        stroke_list = combined_pd["T_Stroke"].tolist()
        char_list = combined_pd["T_Character"].tolist()  # 新增：获取字符列表
    else:
        print("Wrong language!")
        return None, None, None

    # 根据repeat参数决定是否去重
    if repeat == 'deduplicate':
        # 去重模式 - 基于字符去重
        print("使用去重模式(基于字符)...")

        # 记录每个字符第一次出现的位置
        char_dict = {}
        for i, char in enumerate(char_list):
            if char not in char_dict:
                char_dict[char] = i

        # 创建去重后的freq_list和对应的ae_list
        unique_indices = list(char_dict.values())
        processed_freq_list = [freq_list[i] for i in unique_indices]
        processed_stroke_list = [stroke_list[i] for i in unique_indices]

        # 获取去重后的字符列表（可选，用于验证）
        processed_char_list = [char_list[i] for i in unique_indices]

    elif repeat == 'duplicate':
        # 不去重模式
        print("使用不去重模式...")
        processed_freq_list = freq_list
        processed_stroke_list = stroke_list
        processed_char_list = char_list  # 保留完整字符列表

    else:
        print("错误的repeat参数! 应为'duplicate'或'deduplicate'")
        return None, None, None

    # 检查两个列表长度是否一致
    if len(processed_freq_list) != len(processed_stroke_list):
        print(f"错误: 列表长度不一致! freq_list: {len(processed_freq_list)}, ae_list: {len(processed_ae_list)}")
        return None, None, None

    # 计算相关性
    state, correlation, p_value = compute_SPEARMAN(processed_freq_list, processed_stroke_list)

    if p_value < 0.05:
        print("相关性在统计上是显著的。")
        return True, correlation, p_value
    else:
        print("相关性在统计上不显著。")
        return False, correlation, p_value


def Spearman_frequency_ae(model, language, task, mode, type_, repeat):
    inpath = r"combined.xlsx"
    combined_pd = pd.read_excel(inpath)

    if language == 'simple':
        freq_list = combined_pd["S_Freq.log"].tolist()
        char_list = combined_pd["S_Character"].tolist()  # 新增：获取字符列表
    elif language == 'traditional':
        freq_list = combined_pd["T_Freq.log"].tolist()
        char_list = combined_pd["T_Character"].tolist()  # 新增：获取字符列表
    else:
        print("Wrong language!")
        return None, None, None

    # 检查列是否存在
    ae_column_name = f"{model}_{language}_{task}_{mode}_{type_}_ae"
    if ae_column_name not in combined_pd.columns:
        print(f"Column {ae_column_name} does not exist!")
        return None, None, None

    ae_list = combined_pd[ae_column_name].tolist()

    # 根据repeat参数决定是否去重
    if repeat == 'deduplicate':
        # 去重模式 - 基于字符去重
        print("使用去重模式(基于字符)...")

        # 记录每个字符第一次出现的位置
        char_dict = {}
        for i, char in enumerate(char_list):
            if char not in char_dict:
                char_dict[char] = i

        # 创建去重后的freq_list和对应的ae_list
        unique_indices = list(char_dict.values())
        processed_freq_list = [freq_list[i] for i in unique_indices]
        processed_ae_list = [ae_list[i] for i in unique_indices]

        # 获取去重后的字符列表（可选，用于验证）
        processed_char_list = [char_list[i] for i in unique_indices]

    elif repeat == 'duplicate':
        # 不去重模式
        print("使用不去重模式...")
        processed_freq_list = freq_list
        processed_ae_list = ae_list
        processed_char_list = char_list  # 保留完整字符列表

    else:
        print("错误的repeat参数! 应为'duplicate'或'deduplicate'")
        return None, None, None

    # 检查两个列表长度是否一致
    if len(processed_freq_list) != len(processed_ae_list):
        print(f"错误: 列表长度不一致! freq_list: {len(processed_freq_list)}, ae_list: {len(processed_ae_list)}")
        return None, None, None

    # 计算相关性
    state, correlation, p_value = compute_SPEARMAN(processed_freq_list, processed_ae_list)

    if p_value < 0.05:
        print("相关性在统计上是显著的。")
        return True, correlation, p_value
    else:
        print("相关性在统计上不显著。")
        return False, correlation, p_value


def Spearman_stroke_ae(model, language, task, mode, type_, repeat):
    inpath = r"combined.xlsx"
    combined_pd = pd.read_excel(inpath)

    if language == 'simple':
        freq_list = combined_pd["S_Stroke"].tolist()
        char_list = combined_pd["S_Character"].tolist()  # 新增：获取字符列表
    elif language == 'traditional':
        freq_list = combined_pd["T_Stroke"].tolist()
        char_list = combined_pd["T_Character"].tolist()  # 新增：获取字符列表
    else:
        print("Wrong language!")
        return None, None, None

    # 检查列是否存在
    ae_column_name = f"{model}_{language}_{task}_{mode}_{type_}_ae"
    if ae_column_name not in combined_pd.columns:
        print(f"Column {ae_column_name} does not exist!")
        return None, None, None

    ae_list = combined_pd[ae_column_name].tolist()

    # 根据repeat参数决定是否去重
    if repeat == 'deduplicate':
        # 去重模式 - 基于字符去重
        print("使用去重模式(基于字符)...")

        # 记录每个字符第一次出现的位置
        char_dict = {}
        for i, char in enumerate(char_list):
            if char not in char_dict:
                char_dict[char] = i

        # 创建去重后的freq_list和对应的ae_list
        unique_indices = list(char_dict.values())
        processed_freq_list = [freq_list[i] for i in unique_indices]
        processed_ae_list = [ae_list[i] for i in unique_indices]

        # 获取去重后的字符列表（可选，用于验证）
        processed_char_list = [char_list[i] for i in unique_indices]


    elif repeat == 'duplicate':
        # 不去重模式
        print("使用不去重模式...")
        processed_freq_list = freq_list
        processed_ae_list = ae_list
        processed_char_list = char_list  # 保留完整字符列表

    else:
        print("错误的repeat参数! 应为'duplicate'或'deduplicate'")
        return None, None, None

    # 检查两个列表长度是否一致
    if len(processed_freq_list) != len(processed_ae_list):
        print(f"错误: 列表长度不一致! freq_list: {len(processed_freq_list)}, ae_list: {len(processed_ae_list)}")
        return None, None, None

    # 计算相关性
    state, correlation, p_value = compute_SPEARMAN(processed_freq_list, processed_ae_list)

    if p_value < 0.05:
        print("相关性在统计上是显著的。")
        return True, correlation, p_value
    else:
        print("相关性在统计上不显著。")
        return False, correlation, p_value

# 执行Wilcoxon符号秩检验并计算效应量
def wilcoxon_same_different(language, task, model, repeat, alpha=0.05):
    # 读取数据
    same_file_path = r"data1-1_experiment1_same.xlsx"
    same_dic = read_MAEexcel_to_dict(same_file_path, language, task, model_number=5)

    different_file_path = r"data1-1_experiment1_different.xlsx"
    different_dic = read_MAEexcel_to_dict(different_file_path, language, task, model_number=5)

    # 提取数据
    same = list(same_dic[model][repeat]['zero_shot'].values()) + \
           list(same_dic[model][repeat]['few_shot'].values())
    different = list(different_dic[model][repeat]['zero_shot'].values()) + \
                list(different_dic[model][repeat]['few_shot'].values())

    print(f"Same组数据: {same}")
    print(f"Different组数据: {different}")
    print(f"数据长度: {len(same)}")

    # 数据验证
    if len(same) != len(different):
        print("警告: 两组数据长度不一致!")
        return None

    if len(same) == 0:
        print("警告: 数据为空!")
        return None

    # Wilcoxon符号秩检验
    try:
        statistic, p_value = wilcoxon(same, different)
    except Exception as e:
        print(f"Wilcoxon检验出错: {e}")
        return None

    # 计算基本统计量
    differences = np.array(same) - np.array(different)
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # 计算Wilcoxon效应量(r)
    # 计算Z分数
    w_mean = n * (n + 1) / 4
    w_std = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if w_std == 0:
        z = 0
    elif statistic <= w_mean:
        z = (statistic + 0.5 - w_mean) / w_std
    else:
        z = (statistic - 0.5 - w_mean) / w_std

    effect_size_r = np.abs(z) / np.sqrt(n) if n > 0 else 0

    # 解释效应量
    if effect_size_r < 0.1:
        effect_size_interpretation = "可忽略"
    elif effect_size_r < 0.3:
        effect_size_interpretation = "小"
    elif effect_size_r < 0.5:
        effect_size_interpretation = "中"
    else:
        effect_size_interpretation = "大"

    # 计算Cohen's d
    # 确保分母不为零
    if std_diff == 0:
        cohens_d = 0.0
        hedges_g = 0.0
        d_ci_lower = d_ci_upper = 0.0
    else:
        cohens_d = mean_diff / std_diff

        # Hedges' g校正
        j = 1 - (3 / (4 * (n - 1) - 1)) if n > 1 else 1
        hedges_g = cohens_d * j

        # 置信区间
        se_d = np.sqrt((1 / n) + (cohens_d ** 2 / (2 * n)))
        z_crit = norm.ppf(0.975)  # 95%置信区间
        d_ci_lower = cohens_d - z_crit * se_d
        d_ci_upper = cohens_d + z_crit * se_d

    # Cohen's d解释
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        d_interpretation = "可忽略"
    elif abs_d < 0.5:
        d_interpretation = "小"
    elif abs_d < 0.8:
        d_interpretation = "中"
    else:
        d_interpretation = "大"

    # 计算Cliff's Delta
    x = np.array(same)
    y = np.array(different)

    # 计算比较矩阵
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    equal = np.sum(x[:, None] == y[None, :])

    # 计算概率
    total_comparisons = n * n
    p_gt = greater / total_comparisons
    p_lt = less / total_comparisons
    p_eq = equal / total_comparisons

    # Cliff's Delta
    delta = p_gt - p_lt

    # Cliff's Delta置信区间
    se_delta = np.sqrt((p_gt * (1 - p_gt) + p_lt * (1 - p_lt) + 2 * p_gt * p_lt) / (n - 1)) if n > 1 else 0
    z_crit_delta = norm.ppf(0.975)
    delta_ci_lower = max(-1, delta - z_crit_delta * se_delta)
    delta_ci_upper = min(1, delta + z_crit_delta * se_delta)

    # Cliff's Delta解释
    abs_delta = abs(delta)
    if abs_delta < 0.11:
        delta_interpretation = "可忽略"
    elif abs_delta < 0.28:
        delta_interpretation = "小"
    elif abs_delta < 0.43:
        delta_interpretation = "中"
    else:
        delta_interpretation = "大"

    # 返回所有结果
    return {
        'wilcoxon_statistic': statistic,
        'wilcoxon_p_value': p_value,
        'z_score': z,
        'wilcoxon_r': effect_size_r,
        'wilcoxon_r_interpretation': effect_size_interpretation,
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'cohens_d_ci': (d_ci_lower, d_ci_upper),
        'cohens_d_interpretation': d_interpretation,
        'cliffs_delta': delta,
        'cliffs_delta_ci': (delta_ci_lower, delta_ci_upper),
        'cliffs_delta_interpretation': delta_interpretation,
        'p_X_gt_Y': p_gt,
        'p_X_lt_Y': p_lt,
        'p_X_eq_Y': p_eq,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'sample_size': n,
        'is_significant': p_value < alpha,
        'direction': direction
    }


# 简化版本的独立函数（如果需要单独调用）
def calculate_effect_sizes(sample1, sample2):
    # 计算差值
    differences = np.array(sample1) - np.array(sample2)
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # Cohen's d
    cohens_d = mean_diff / std_diff if std_diff != 0 else 0

    # Hedges' g校正
    j = 1 - (3 / (4 * (n - 1) - 1)) if n > 1 else 1
    hedges_g = cohens_d * j

    # Cliff's Delta
    x, y = np.array(sample1), np.array(sample2)
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    equal = np.sum(x[:, None] == y[None, :])

    total_comparisons = n * n
    p_gt = greater / total_comparisons
    p_lt = less / total_comparisons
    p_eq = equal / total_comparisons
    delta = p_gt - p_lt

    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'cliffs_delta': delta,
        'p_gt': p_gt,
        'p_lt': p_lt,
        'p_eq': p_eq,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'n': n
    }

# 执行Wilcoxon符号秩检验并计算效应量
def wilcoxon_test_with_effect(same, different, alpha=0.05):
    # 确保两组数据长度相同
    assert len(same) == len(different), "两组数据长度必须相同"

    n = len(same)  # 样本量
    print(f"样本量: n = {n}")

    # 进行Wilcoxon符号秩检验（使用渐近近似，适合n>=25）
    # method='approx' 使用正态近似，适合中等样本量
    statistic, p_value = wilcoxon(same, different) #method='approx'

    print(f"Wilcoxon W统计量: {statistic}")
    print(f"P值: {p_value:.6f}")

    # 计算差值
    differences = np.array(same) - np.array(different)

    # 手动计算Z分数和效应量r（更准确）
    # 排除差值为0的配对
    nonzero_diffs = differences[differences != 0]
    n_nonzero = len(nonzero_diffs)

    if n_nonzero == 0:
        print("警告: 所有差值都为0，无法计算Z分数和效应量")
        z = 0
        effect_size_r = 0
    else:
        # 计算非零差值的绝对值的秩
        abs_diffs = np.abs(nonzero_diffs)
        ranks = rankdata(abs_diffs)

        # 计算正秩和和负秩和
        positive_ranks = np.sum(ranks[nonzero_diffs > 0])
        negative_ranks = np.sum(ranks[nonzero_diffs < 0])

        # Wilcoxon统计量是正秩和和负秩和中的较小者
        # 注意：scipy返回的statistic可能不是这个值，但为了计算Z，我们使用这个定义
        W = min(positive_ranks, negative_ranks)

        # 计算Z分数（连续性校正）
        w_mean = n_nonzero * (n_nonzero + 1) / 4
        w_std = np.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24)

        if w_std == 0:
            z = 0
        elif W <= w_mean:
            z = (W + 0.5 - w_mean) / w_std
        else:
            z = (W - 0.5 - w_mean) / w_std

        # 计算效应量r（配对样本的效应量）
        effect_size_r = np.abs(z) / np.sqrt(n_nonzero)

    # 效应量解释
    if effect_size_r < 0.1:
        effect_size_interpretation = "可忽略"
    elif effect_size_r < 0.3:
        effect_size_interpretation = "小"
    elif effect_size_r < 0.5:
        effect_size_interpretation = "中"
    else:
        effect_size_interpretation = "大"

    # 计算描述性统计
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    median_diff = np.median(differences)

    # 返回结果
    return {
        'statistic': statistic,
        'p_value': p_value,
        'z_score': z,
        'effect_size_r': effect_size_r,
        'effect_interpretation': effect_size_interpretation,
        'mean_difference': mean_diff,
        'median_difference': median_diff,
        'std_difference': std_diff,
        'n': n,
        'n_nonzero': n_nonzero,
        'is_significant': p_value < alpha
    }

def compute_simple_traditional_correlation():
    inpath = r"combined.xlsx"
    combined_pd = pd.read_excel(inpath)

    sp_aoa_list = combined_pd["S_AoA"].tolist()
    print(sp_aoa_list)
    sp_fam_list = combined_pd["S_Familiarity"].tolist()
    print(sp_fam_list)
    sp_con_list = combined_pd["S_Concreteness"].tolist()
    print(sp_con_list)
    sp_ima_list = combined_pd["S_Imageability"].tolist()
    print(sp_ima_list)

    td_aoa_list = combined_pd["T_AoA"].tolist()
    print(td_aoa_list)
    td_fam_list = combined_pd["T_Familiarity"].tolist()
    print(td_fam_list)
    td_con_list = combined_pd["T_Concreteness"].tolist()
    print(td_con_list)
    td_ima_list = combined_pd["T_Imageability"].tolist()
    print(td_ima_list)

    aoa = compute_SPEARMAN(sp_aoa_list, td_aoa_list)
    fam = compute_SPEARMAN(sp_fam_list, td_fam_list)
    con = compute_SPEARMAN(sp_con_list, td_con_list)
    ima = compute_SPEARMAN(sp_ima_list, td_ima_list)

    print(f'aoa: {aoa}')
    print(f'fam: {fam}')
    print(f'con: {con}')
    print(f'ima: {ima}')

def wilcoxon_sum_rank(language, task, mode, _type, model, repeat, alpha=0.05):

    column_name = f"{model}_{language}_{task}_{mode}_{_type}_ae"

    same_path = r"combined_e1_same.xlsx"
    different_path = r"combined_e1_different.xlsx"

    raw_same_pd = pd.read_excel(same_path)
    raw_different_pd = pd.read_excel(different_path)

    same_score_list = raw_same_pd[column_name]
    different_score_list = raw_different_pd[column_name]

    print(same_score_list)
    print(different_score_list)

    # 基本统计
    # 执行Mann-Whitney U检验并生成完整报告
    n_a, n_b = len(same_score_list), len(different_score_list)
    median_a, median_b = np.median(same_score_list), np.median(different_score_list)
    q1_a, q3_a = np.percentile(same_score_list, [25, 75])
    q1_b, q3_b = np.percentile(different_score_list, [25, 75])
    iqr_a, iqr_b = q3_a - q1_a, q3_b - q1_b

    # 执行检验
    # 计算Mann-Whitney U检验
    u_stat, p_value = mannwhitneyu(same_score_list, different_score_list, alternative='two-sided')
    u2 = n_a * n_b - u_stat  # 另一个U值
    u_min = min(u_stat, u2)

    # 计算W统计量
    w1 = u_stat + n_a * (n_a + 1) / 2
    w2 = (n_a * n_b) - u_stat + n_b * (n_b + 1) / 2

    # 计算效应量
    # Z分数
    mean_u = n_a * n_b / 2
    std_u = np.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

    # 连续性校正
    if u_stat <= mean_u:
        z = (u_stat + 0.5 - mean_u) / std_u
    else:
        z = (u_stat - 0.5 - mean_u) / std_u

    # r效应量
    r_effect = np.abs(z) / np.sqrt(n_a + n_b)

    # 4. 效应量解释
    def interpret_effect_size(r):
        if r < 0.1:
            return "可忽略"
        elif r < 0.3:
            return "小"
        elif r < 0.5:
            return "中"
        else:
            return "大"

    effect_interpretation = interpret_effect_size(r_effect)

    # 方向判断
    direction = ""
    if median_a > median_b:
        direction = f"same > different"
    elif median_a < median_b:
        direction = f"same < different"
    else:
        direction = "无差异"

    # 显著性判断
    significant = p_value < alpha

    # 返回完整结果
    results = {
        # 描述统计
        'n1': n_a,
        'n2': n_b,
        'median1': median_a,
        'median2': median_b,
        'iqr1': iqr_a,
        'iqr2': iqr_b,
        'q1_1': q1_a,
        'q3_1': q3_a,
        'q1_2': q1_b,
        'q3_2': q3_b,

        # 检验统计
        'U': u_stat,
        'U_min': u_min,
        'w1': w1,
        'w2': w2,
        'p_value': p_value,
        'z_score': z,
        'significant': significant,

        # 效应量
        'r_effect': r_effect,
        'r_interpretation': effect_interpretation,

        # 其他
        'direction': direction,
        'alpha': alpha
    }

    return results


if __name__ == '__main__':
    pass
