# Cantonese_norms
<!-- Copyright [19 Mar 2024] [florethsong]  

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.-->

The data and code for exploring "Can Large Language Models Help to Collect Psycholinguistic Norms in Low-resource Languages?"

We made a case study on Cantonese traditional characters with comparisons to Mandarin simplified characters across four psycholinguistic variables: Age-of-Acquisition, Familiarity, Concreteness, and Imageability.

---
:pushpin:**Reference for Cantonese Norms**: 

Su, I. F., Yum, Y. N., & Lau, D. K. Y. (2023). Hong Kong Chinese character psycholinguistic norms: Ratings of 4376 single Chinese characters on semantic radical transparency, age-of-acquisition, familiarity, imageability, and concreteness. Behavior research methods, 55(6), 2989-3008.
  
:pushpin:**Reference for Mandarin Morms**: 

Liu, Y., Shu, H., & Li, P. (2007). Word naming and psycholinguistic norms: Chinese. Behavior research methods, 39(2), 192-198.


### :card_index_dividers: Description of file structure
```
.
│  
├─code
│      data.py # code for clearing and adjusting data format and content
│      model.py # code for accessing API of target LLMs
│      prompt.py # code for compiling different prompts
│      statistics.py # code for computing statistical test
│      visualization.py # code for visualizing the results data
│      
└─data
    ├─norms
    │      @merged.xlsx # curated test data merged from original Cantonese and Chinese norms
    │      
    └─results
        ├─experiment1
        │      combined.xlsx # results of multiple LLMs (i.e., GPT-4.1 mini, Llama-3-70B-Instruct, Mistral-Small-3.2-24B-Instruct, DeepSeek-V3, Qwen2.5-72B-Instruct) across different variables and prompt settings
        │      
        ├─experiment2
        │      combined.xlsx results of Qwen2.5-7B-Instruct and CantoneseLLMChat-7B across different variables and prompt settings
        │      
        └─experiment_same+different
                @TS_different.xlsx # characters with different scripts across Chinese varieties.
                @TS_same.xlsx # characters with same scripts across Chinese varieties.
                combined_different.xlsx # results of LLMs in experiments1 on the different dataset.
                combined_same.xlsx # results of LLMs in experiments1 on the same dataset.
```

## Citation
Please cite the following paper if it is helpful to your work :)!
```

```
