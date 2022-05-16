# Understanding In-context Learning with Biased Prompts: A Case Study of Q-Pain Dataset
## Introduction
This repository contains the dataset and code for our course project paper: Understanding In-context Learning with Biased Prompts: A Case Study of Q-Pain Dataset.
## Biased Prompts and Inference Results
`Q_Pain.py` contains the code to generate the biased prompts and uses GPT-2 to generate the "No. Probability" for each medical context.
There are two arguments for `Q_Pain.py`. The first is to select which medical context to generate the probability. The second is which prompts
we choose to use. `--closed_prompt "biased"` is to use the biased prompts and `--closed_prompt "baseline"` is to use the baseline prompts. 
`Q_Pain.py` can be run with the following command:

```bash
python Q_Pain.py --medical_context_file "data_acute_cancer.csv" --closed_prompt "biased"
```
`t-test.py` contains the code to calculate the p-value for the significance test.

`heatmap_plot.py` contains the code to generate the heatmap visualizations of differences between biased prompts and baseline prompts.

`representative_bias.py` and `representative_analysis.ipynb` contains the code to examine the representative bias.

`results` folder contains the original Q-pain dataset and the heatmap visualizations.
`iterated_results` folder contains the "No. probability" of each demographic combinations in prompts.

`reddit_results` contains the code the script to attempt to generate additional data to supplement Q-pain dataset, 
but we do not get suitable dataset.

`Q_Pain_Experiments.ipynb` is the code for us to experiment how to write the code and will not generate our main results.

Please refer to `requirements.txt` file to install the packages and we run our code with Python3.7.



