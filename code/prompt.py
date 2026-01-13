# -*- coding:utf-8 -*-
'''
@file   :prompt.py
@author :Floret Huacheng SONG
@time   :18/11/2025 上午10:44
@purpose: for compiling different prompts
'''

# 7-pointLikertscale

# reference:
# Barca, L., Burani, C., & Arduino, L. S. (2002). Word naming times and psycholinguistic norms for Italian nouns. Behavior research methods, instruments, & computers, 34(3), 424-434.
# Liu, Y., Shu, H., & Li, P. (2007). Word naming and psycholinguistic norms: Chinese. Behavior research methods, 39(2), 192-198.
# Su, I. F., Yum, Y. N., & Lau, D. K. Y. (2023). Hong Kong Chinese character psycholinguistic norms: Ratings of 4376 single Chinese characters on semantic radical transparency, age-of-acquisition, familiarity, imageability, and concreteness. Behavior research methods, 55(6), 2989-3008.
# Martínez, G., Conde, J., Reviriego, P., & Brysbaert, M. (2025). AI-generated estimates of familiarity, concreteness, valence, and arousal for over 100,000 Spanish words. Quarterly Journal of Experimental Psychology, 78(10), 2272-2283.
# Martínez, G., Molero, J. D., González, S., Conde, J., Brysbaert, M., & Reviriego, P. (2024). Using large language models to estimate features of multi-word expressions: Concreteness, valence, arousal. Behavior Research Methods, 57(1), 5.
# Conde, J., Grandury, M., Fu, T., Arriaga, C., Martínez, G., Clark, T., ... & Brysbaert, M. (2025). Adding LLMs to the psycholinguistic norming toolbox: A practical guide to getting the most out of human ratings. arXiv preprint arXiv:2509.14405.

system_en = 'You are a language expert serving for evaluating psycholinguistic properties of the given characters.'

system_zh = '你是一位语言专家，负责评估给定汉字的心理语言学属性。'

system_ca = '你係一位語言專家，負責評估給定漢字嘅心理語言學屬性。'


## ZERO-SHOT
# study 1: AoA

zeroprompt1_en = '''
** Instructions **
Age of acquisition (AoA) is defined as the estimated age at which the meaning of a character or a word was first learned by a learner, either in the auditory or in written modality.
A character or a word with an early time-point of AoA can be recognised and read faster and more accurately than the later-acquired ones.

Please rate the most possible AoA for the following given character on a scale from 1 to 7, where the number represents the range of first acquired ages, i.e., 1 means 0-2 years, 2means 3-4 years, 3 means 5–6 years, 4 means 7–8 years, 5 means 9–10 years, 6 means 11-12 years, and 7 means over 13 years.

Please limit your answer to a valid JSON object in the following format: {"Character": given character, "AoA": Your judgement of AoA of the given character as an integer from 1 to 7.}

** Input (This is a [] character.) **
{"Character": "{}"}

** Output **

'''

zeroprompt1_zh = '''
** 指令 **
习得年龄（AoA）是指学习者首次理解某个汉字或词语含义的预估年龄，这种习得可能通过听觉或书面形式实现。
习得时间较早的汉字或词语，其识别和阅读速度会更快，准确率也高于习得时间较晚的汉字或词语。

请按1-7的尺度评估以下给定汉字最可能的习得年龄，其中数值代表首次习得该汉字的年龄范围，即：1表示0-2岁，2表示3-4岁，3表示5–6岁，4表示7–8岁，5表示9–10岁，6表示11-12岁，7表示13岁以上。

请以如下的有效JSON格式输出你的答案：{"Character": 给定汉字, "AoA": 你对给定汉字习得年龄的评分（1-7之间的整数）}

** 输入（这是一个[]汉字。）**
{"Character": "{}"}

** 输出 **

'''

zeroprompt1_ca = '''
** 指令 **
習得年齡（AoA）係指學習者首次理解某個漢字或詞語含義嘅預估年齡，呢種習得可能透過聽覺或書面形式實現。
習得時間較早嘅漢字或詞語，其識別同閱讀速度會更快，準確率亦高過習得時間較晚嘅漢字或詞語。

請按1-7嘅尺度評估以下給定漢字最可能嘅習得年齡，其中數值代表首次習得該漢字嘅年齡範圍，即：1表示0-2歲，2表示3-4歲，3表示5–6歲，4表示7–8歲，5表示9–10歲，6表示11-12歲，7表示13歲以上。

請以如下有效嘅JSON格式輸出你嘅答案：{"Character": 給定漢字, "AoA": 你對給定漢字習得年齡嘅評分（1-7之間嘅整數）}

** 輸入（呢個係[]漢字。）**
{"Character": "{}"}

** 輸出 **

'''

# study 2: Imageability

zeroprompt2_en = '''
** Instructions **
Imageability is a measure of the ease and speed with which a character or a word evokes the association of a mental image, a visual representation, a sound, or any other sensory experience. 
A character or a word with a higher degree of imageability tends to arouse a sensory association more easily and more quickly.

Please rate the imageability for the following given character on a scale from 1 to 7, where 1 means low imageability and 7 means high imageability with the midpoint representing moderate imageability.

Please limit your answer to a valid JSON object in the following format: {"Character": given character, "Imageability": Your judgement of imageability of the given character as an integer from 1 to 7.}
** Input (This is a [] character.) **
{"Character": "{}"}

** Output **

'''

zeroprompt2_zh = '''
** 指令 **
表象性是指一个汉字或词语激发心理意象、视觉表征、声音或任何其他感官体验的联想的难易程度和速度。
表象性更高的汉字或词语能更容易和更快速地唤起感官联想。

请按1-7的尺度评估以下给定汉字的表象性程度，其中1代表低表象性，7代表高表象性，中点表示中等表象性。

请以如下的有效JSON格式输出你的答案：{"Character": 给定汉字, "Imageability": 你对给定汉字表象性的评分（1-7之间的整数）}

** 输入（这是一个[]汉字。）**
{"Character": "{}"}

** 输出 **

'''

zeroprompt2_ca = '''
** 指令 **
表象性係指一個漢字或詞語激發心理意象、視覺表徵、聲音或任何其他感官體驗聯想嘅難易程度同速度。
表象性更高嘅漢字或詞語能夠更加容易同快速咁引發感官聯想。

請按1-7嘅尺度評估以下給定漢字嘅表象性程度，其中1代表低表象性，7代表高表象性，中點表示中等表象性。

請以如下有效嘅JSON格式輸出你嘅答案：{"Character": 給定漢字, "Imageability": 你對給定漢字表象性嘅評分（1-7之間嘅整數）}

** 輸入（呢個係[]漢字。）**
{"Character": "{}"}

** 輸出 **

'''

# study 3: Concreteness

zeroprompt3_en = '''
** Instructions **
Concreteness is a measure of the property of a character or a word to refer to physical objects,animate beings,actions, or materials that can be experienced directly by the senses.
A character or a word with a higher degree of concreteness tends to represent something that more likely exists in a definite physical form in the real world rather than representing more of an abstract concept or idea.

Please rate the concreteness for the following given character on a scale from 1 to 7, where 1 means low concreteness and 7 means high concreteness with the midpoint representing moderate concreteness.

Please limit your answer to a valid JSON object in the following format: {"Character": given character, "Concreteness": Your judgement of concreteness of the given character as an integer from 1 to 7.}

** Input (This is a [] character.) **
{"Character": "{}"}

** Output **

'''

zeroprompt3_zh = '''
** 指令 **
具体性是指一个汉字或词语指代物理实体、生命体、行为或能够被感官直接体验的材料的属性。
具体性更高的汉字或词语更可能代表现实中以明确物理形态存在的事物，而非代表抽象的概念或观念。

请按1-7的尺度评估以下给定汉字的具体性程度，其中1代表低具体性，7代表高具体性，中点表示中等具体性。

请以如下的有效JSON格式输出你的答案：{"Character": 给定汉字, "Concreteness": 你对给定汉字具体性的评分（1-7之间的整数）}

** 输入（这是一个[]汉字。）**
{"Character": "{}"}

** 输出 **

'''

zeroprompt3_ca = '''
** 指令 **
具體性係指一個漢字或詞語指代物理實體、生命體、行為或者能夠被感官直接體驗嘅材料嘅屬性。
具體性更高嘅漢字或詞語更可能代表現實中以明確物理形態存在嘅事物，而唔係代表抽象嘅概念或者觀念。

請按1-7嘅尺度評估以下給定漢字嘅具體性程度，其中1代表低具體性，7代表高具體性，中點表示中等具體性。

請以如下有效嘅JSON格式輸出你嘅答案：{"Character": 給定漢字, "Concreteness": 你對給定漢字具體性嘅評分（1-7之間嘅整數）}

** 輸入（呢個係[]漢字。）**
{"Character": "{}"}

** 輸出 **

'''

# study 4: Familiarity

zeroprompt4_en = '''
** Instructions **
Familiarity is defined as how commonly a character or a word is used and encountered, based on its estimated frequency in written or spoken language.
A character or a word that appears more frequently in daily life tends to feel more familiar and are recognized more easily.

Please rate the familiarity for the following given character on a scale from 1 to 7, where 1 means low familiarity and 7 means high familiarity with the midpoint representing moderate familiarity.

Please limit your answer to a valid JSON object in the following format: {"Character": given character, "Familiarity": Your judgement of familiarity of the given character as an integer from 1 to 7.}

** Input (This is a [] character.) **
{"Character": "{}"}

** Output **

'''

zeroprompt4_zh = '''
** 指令 **
熟悉度是指一个汉字或词语在书面或口语中的预估出现频率，反映其被使用和接触的普遍程度。
在日常生活中出现频率越高的汉字或词语，通常感觉越熟悉，也更容易被识别。

请按1-7的尺度评估以下给定汉字的熟悉度，其中1代表低熟悉度，7代表高熟悉度，中点表示中等熟悉度。

请以如下的有效JSON格式输出你的答案：{"Character": 给定汉字, "Familiarity": 你对给定汉字熟悉度的评分（1-7之间的整数）}

** 输入（这是一个[]汉字。）**
{"Character": "{}"}

** 输出 **

'''

zeroprompt4_ca = '''
** 指令 **
熟悉度係指一個漢字或詞語喺書面或口語中使用嘅預估頻率，反映佢被接觸嘅普遍程度。
日常生活中出現頻率越高嘅漢字或詞語，通常感覺越熟悉，亦都更容易被識別。

請按1-7嘅尺度評估以下給定漢字嘅熟悉度，其中1代表低熟悉度，7代表高熟悉度，中點表示中等熟悉度。

請以如下有效嘅JSON格式輸出你嘅答案：{"Character": 給定漢字, "Familiarity": 你對給定漢字熟悉度嘅評分（1-7之間嘅整數）}

** 輸入（呢個係[]漢字。）**
{"Character": "{}"}

** 輸出 **

'''


## FEW-SHOT
# study 1: AoA

fewprompt1_en = '''
** Instructions **
Age of acquisition (AoA) is defined as the estimated age at which the meaning of a character or a word was first learned by a learner, either in the auditory or in written modality.
A character or a word with an early time-point of AoA can be recognised and read faster and more accurately than the later-acquired ones.

Please rate the most possible AoA for the following given character on a scale from 1 to 7, where the number represents the range of first acquired ages, i.e., 1 means 0-2 years, 2means 3-4 years, 3 means 5–6 years, 4 means 7–8 years, 5 means 9–10 years, 6 means 11-12 years, and 7 means over 13 years.

- Examples of characters that would get a rating of 1 are "一", "二" and "十". 
- Examples of characters that would get a rating of 7 are "洳", "酞" and "觔".

Please limit your answer to a valid JSON object in the following format: {"Character": given character, "AoA": Your judgement of AoA of the given character as an integer from 1 to 7.}

** Input (This is a [] character.) **
{"Character": "{}"}

** Output **

'''

fewprompt1_zh ='''：
** 指令 **
习得年龄（AoA）是指学习者首次理解某个汉字或词语含义的预估年龄，这种习得可能通过听觉或书面形式实现。
习得时间较早的汉字或词语，其识别和阅读速度会更快，准确率也高于习得时间较晚的汉字或词语。

请按1-7的尺度评估以下给定汉字最可能的习得年龄，其中数值代表首次习得该汉字的年龄范围，即：1表示0-2岁，2表示3-4岁，3表示5–6岁，4表示7–8岁，5表示9–10岁，6表示11-12岁，7表示13岁以上。

- 可获评1分的汉字例子有：“一”，“二”，“十”；
- 可获评7分的汉字例子有：“洳”，“酞”，“觔”。

请以如下的有效JSON格式输出你的答案：{"Character": 给定汉字, "AoA": 你对给定汉字习得年龄的评分（1-7的整数）}

** 输入（这是一个[]汉字。）**
{"Character": "{}"}

** 输出 **

'''

fewprompt1_ca = '''
** 指令 **
習得年齡（AoA）係指學習者首次理解某個漢字或詞語含義嘅預估年齡，呢種習得可能透過聽覺或書面形式實現。
習得時間較早嘅漢字或詞語，其識別同閱讀速度會更快，準確率亦高過習得時間較晚嘅漢字或詞語。

請按1-7嘅尺度評估以下給定漢字最可能嘅習得年齡，其中數值代表首次習得該漢字嘅年齡範圍，即：1表示0-2歲，2表示3-4歲，3表示5–6歲，4表示7–8歲，5表示9–10歲，6表示11-12歲，7表示13歲以上。

- 可獲評1分嘅漢字例子有：「一」，「二」，「十」;
- 可獲評7分嘅漢字例子有：「洳」，「酞」，「觔」。

請以如下有效嘅JSON格式輸出你嘅答案：{"Character": 給定漢字, "AoA": 你對給定漢字習得年齡嘅評分（1-7嘅整數）}

** 輸入（呢個係[]漢字。）**
{"Character": "{}"}

** 輸出 **

'''

# study 2: Imageability

fewprompt2_en = '''
** Instructions **
Imageability is a measure of the ease and speed with which a character or a word evokes the association of a mental image, a visual representation, a sound, or any other sensory experience. 
A character or a word with a higher degree of imageability tends to arouse a sensory association more easily and more quickly.

Please rate the imageability for the following given character on a scale from 1 to 7, where 1 means low imageability and 7 means high imageability with the midpoint representing moderate imageability.

- Examples that would get a rating of 1 are "倣", "琤" and "澂". 
- Examples that would get a rating of 7 are "米", "象" and "天".

Please limit your answer to a valid JSON object in the following format: 
{"Character": given character, "Imageability": Your judgement of imageability of the given character as an integer from 1 to 7.}

** Input (This is a [] character.) **
{"Character": "{}"}

** Output **

'''

fewprompt2_zh = '''
** 指令 **
表象性是指一个汉字或词语激发心理意象、视觉表征、声音或任何其他感官体验的联想的难易程度和速度。
表象性更高的汉字或词语能更容易和更快速地唤起感官联想。

请按1-7的尺度评估以下给定汉字的表象性程度，其中1代表低表象性，7代表高表象性，中点表示中等表象性。

- 可获评1分的汉字例子有：“倣”，“琤”，“澂”；
- 可获评7分的汉字例子有：“米”，“象”，“天”。

请以如下的有效JSON格式输出你的答案：{"Character": 给定汉字, "Imageability": 你对给定汉字表象性的评分（1-7之间的整数）}

** 输入（这是一个[]汉字。）**
{"Character": "{}"}

** 输出 **

'''

fewprompt2_ca = '''
** 指令 **
表象性係指一個漢字或詞語激發心理意象、視覺表徵、聲音或任何其他感官體驗聯想嘅難易程度同速度。
表象性更高嘅漢字或詞語能夠更加容易同快速咁引發感官聯想。

請按1-7嘅尺度評估以下給定漢字嘅表象性程度，其中1代表低表象性，7代表高表象性，中點表示中等表象性。

- 可獲評1分嘅漢字例子有：「倣」，「琤」，「澂」;
- 可獲評7分嘅漢字例子有：「米」，「象」，「天」。

請以如下有效嘅JSON格式輸出你嘅答案：{"Character": 給定漢字, "Imageability": 你對給定漢字表象性嘅評分（1-7之間嘅整數）}

** 輸入（呢個係[]漢字。）**
{"Character": "{}"}

** 輸出 **

'''

# study 3: Concreteness

fewprompt3_en = '''
** Instructions **
Concreteness is a measure of the property of a character or a word to refer to physical objects,animate beings,actions, or materials that can be experienced directly by the senses.
A character or a word with a higher degree of concreteness tends to represent something that more likely exists in a definite physical form in the real world rather than representing more of an abstract concept or idea.

Please rate the concreteness for the following given character on a scale from 1 to 7, where 1 means low concreteness and 7 means high concreteness with the midpoint representing moderate concreteness.

- Examples that would get a rating of 1 are "嘅", "薛" and "郴". 
- Examples that would get a rating of 7 are "鯨", "杯" and "橙".

Please limit your answer to a valid JSON object in the following format: 
{"Character": given character, "Concreteness": Your judgement of concreteness of the given character as an integer from 1 to 7.}

** Input (This is a [] character.) **
{"Character": "{}"}

** Output **

'''

fewprompt3_zh = '''
** 指令 **
具体性是指一个汉字或词语指代物理实体、生命体、行为或能够被感官直接体验的材料的属性。
具体性更高的汉字或词语更可能代表现实中以明确物理形态存在的事物，而非代表抽象的概念或观念。

请按1-7的尺度评估以下给定汉字的具体性程度，其中1代表低具体性，7代表高具体性，中点表示中等具体性。

- 可获评1分的汉字例子有：“嘅”，“薛”，“郴”；
- 可获评7分的汉字例子有：“鯨”，“杯”，“橙”。

请以如下的有效JSON格式输出你的答案：{"Character": 给定汉字, "Concreteness": 你对给定汉字具体性的评分（1-7之间的整数）}

** 输入（这是一个[]汉字。）**
{"Character": "{}"}

** 输出 **

'''

fewprompt3_ca = '''
** 指令 **
具體性係指一個漢字或詞語指代物理實體、生命體、行為或者能夠被感官直接體驗嘅材料嘅屬性。
具體性更高嘅漢字或詞語更可能代表現實中以明確物理形態存在嘅事物，而唔係代表抽象嘅概念或者觀念。

請按1-7嘅尺度評估以下給定漢字嘅具體性程度，其中1代表低具體性，7代表高具體性，中點表示中等具體性。

- 可獲評1分嘅漢字例子有：「嘅」，「薛」，「郴」;
- 可獲評7分嘅漢字例子有：「鯨」，「杯」，「橙」。

請以如下有效嘅JSON格式輸出你嘅答案：{"Character": 給定漢字, "Concreteness": 你對給定漢字具體性嘅評分（1-7之間嘅整數）}

** 輸入（呢個係[]漢字。）**
{"Character": "{}"}

** 輸出 **

'''

# study 4: Familiarity

fewprompt4_en = '''
** Instructions **
Familiarity is defined as how commonly a character or a word is used and encountered, based on its estimated frequency in written or spoken language.
A character or a word that appears more frequently in daily life tends to feel more familiar and are recognized more easily.

Please rate the familiarity for the following given character on a scale from 1 to 7, where 1 means low familiarity and 7 means high familiarity with the midpoint representing moderate familiarity.

- Examples that would get a rating of 1 are "弢", "舁" and "堃". 
- Examples that would get a rating of 7 are "日", "我" and "子".

Please limit your answer to a valid JSON object in the following format: 
{"Character": given character, "Familiarity": Your judgement of familiarity of the given character as an integer from 1 to 7.}

** Input (This is a [] character.) **
{"Character": "{}"}

** Output **

'''

fewprompt4_zh = '''
** 指令 **
熟悉度是指一个汉字或词语在书面或口语中的预估出现频率，反映其被使用和接触的普遍程度。
在日常生活中出现频率越高的汉字或词语，通常感觉越熟悉，也更容易被识别。

请按1-7的尺度评估以下给定汉字的熟悉度，其中1代表低熟悉度，7代表高熟悉度，中点表示中等熟悉度。

- 可获评1分的汉字例子有：“弢”，“舁”，“堃”；
- 可获评7分的汉字例子有：“日”，“我”，“子”。

请以如下的有效JSON格式输出你的答案：{"Character": 给定汉字, "Familiarity": 你对给定汉字熟悉度的评分（1-7之间的整数）}

** 输入（这是一个[]汉字。）**
{"Character": "{}"}

** 输出 **

'''

fewprompt4_ca = '''
** 指令 **
熟悉度係指一個漢字或詞語喺書面或口語中使用嘅預估頻率，反映佢被接觸嘅普遍程度。
日常生活中出現頻率越高嘅漢字或詞語，通常感覺越熟悉，亦都更容易被識別。

請按1-7嘅尺度評估以下給定漢字嘅熟悉度，其中1代表低熟悉度，7代表高熟悉度，中點表示中等熟悉度。

- 可獲評1分嘅漢字例子有：「弢」，「舁」，「堃」;
- 可獲評7分嘅漢字例子有：「日」，「我」，「子」。

請以如下有效嘅JSON格式輸出你嘅答案：{"Character": 給定漢字, "Familiarity": 你對給定漢字熟悉度嘅評分（1-7之間嘅整數）}

** 輸入（呢個係[]漢字。）**
{"Character": "{}"}

** 輸出 **

'''

def prompt_selection(language, type, task, mode, input):
    """
    type: 'en', 'zh', or 'ca'
    task: 'AoA', 'imageability', 'concreteness', 'familiarity'
    mode: 'zero' (zero-shot) or 'few' (few-shot)
    input: the actual character to evaluate
    """

    # Map task names to prompt indices
    task_map = {
        "AoA": 1,
        "imageability": 2,
        "concreteness": 3,
        "familiarity": 4
    }

    # Validate task
    if task not in task_map:
        raise ValueError("Invalid task. Choose from: AoA, imageability, concreteness, familiarity.")

    # Validate type
    if type not in ["en", "zh", "ca"]:
        raise ValueError("Invalid type. Choose from: en, zh, ca.")

    # Validate mode
    if mode not in ["zero", "few"]:
        raise ValueError("Invalid mode. Choose 'zero' or 'few'.")

    index = task_map[task]

    # Build variable name to fetch the prompt
    if mode == "zero":
        var_name = f"zeroprompt{index}_{type}"
    else:
        var_name = f"fewprompt{index}_{type}"

    # Retrieve the prompt from global variables
    if var_name not in globals():
        raise ValueError(f"Prompt {var_name} not found.")

    system_name = f"system_{type}"
    system_prompt = globals()[system_name]

    prompt_template = globals()[var_name]

    if language == 'simple':
        if type == 'en':
            prompt_filled = prompt_template.replace("[]", "Mandarin")
        elif type == 'zh':
            prompt_filled = prompt_template.replace("[]", "普通话")
        elif type == 'ca':
            prompt_filled = prompt_template.replace("[]", "普通話")
    elif language == 'traditional':
        if type == 'en':
            prompt_filled = prompt_template.replace("[]", "Cantonese")
        elif type == 'zh':
            prompt_filled = prompt_template.replace("[]", "粤语")
        elif type == 'ca':
            prompt_filled = prompt_template.replace("[]", "粵語")
    else:
        print("Wrong language")

    user_prompt = prompt_filled.replace("{}", input)

    return system_prompt, user_prompt


if __name__ == '__main__':
    pass
