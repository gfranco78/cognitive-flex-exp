*Prompt Engineering for Cognitive Flexibility

This repository contains the source code used to run mini-expirements examining the concept of cognitive flexibility in LLMs as described in this article. It leverages the new MMLU pro dataset which you can find here https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

The csv 'experiment-results-article-referenced' contains the complete results referenced in the article.

The script 'cognitive_flex_exp.py' is the main script and provides control over the number of samples for each question category. There are also functions to control which prompt templates to run as well as the total number of questions.

The script 'count_tokens.py' is an optional post-processing script which takes the results produced by the main script and counts the tokens in the LLM answers.
