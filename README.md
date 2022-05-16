# Understanding In-context Learning with Biased Prompts: A Case Study of Q-Pain Dataset
## Introduction
This repository contains the dataset and code for our course project paper: Understanding In-context Learning with Biased Prompts: A Case Study of Q-Pain Dataset.
## Biased Prompts and Inference Results
`Q_Pain.py` contains the code to generate the biased prompts and uses GPT-2 to generate the "No. Probability" for each medical context.
There are two arguments for `Q_Pain.py`. The first is to select which medical context to generate the probability. The second is which prompts
we choose to use. `--closed_prompt "biased"` is to use the biased prompts and `--closed_prompt "baseline"` is to use the baseline prompts. 

```bash
python Q_Pain.py --medical_context_file "data_acute_cancer.csv" --closed_prompt "biased"
```
`t-test.py` is the code to calculate the p-value for the significance test.

`bar_plot.py` is the code to generated the heatmap visualizations of differences between biased prompts and baseline prompts.

`results` folder contains the probability of each demographic combinations and the heatmap visualizations.




