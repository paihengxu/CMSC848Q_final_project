import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Draw the figure of different demographic combination in open prompt
context_lists = ["acute_cancer", "acute_non_cancer", "chronic_cancer", "chronic_non_cancer", "post_op"]
for medical_context in context_lists:
    # medical_context = "acute_cancer"

    t_test_result = pd.read_csv('t-test-result.csv')
    medical_context_result = t_test_result[t_test_result['context'] == medical_context]
    # print(medical_context_result)

    result = []
    open_prompt_demo = []
    closed_prompt_demo = []

    for index, row in medical_context_result.iterrows():
        # closed_prompt_race, closed_prompt_gender, open_prompt_race, open_prompt_gender, group_mean_diff
        result.append(-row['group_mean_diff'] * 100)
        open_prompt_demo.append(row['open_prompt_race'][0] + row['open_prompt_gender'][0].upper())
        closed_prompt_demo.append(row['closed_prompt_race'][0] + row['closed_prompt_gender'][0].upper())

    result_df = pd.DataFrame(
        {'No. Probability Difference': result,
         'Demographic Combination in Open Prompts': open_prompt_demo,
         'Demographic Combination in Closed Prompts': closed_prompt_demo
         })

    sns.set(rc={'figure.figsize': (8.5, 4.5)})
    # sns.set_style("white")
    # plt.ylim(0, 39)
    # sns.set(font_scale=1.3)
    sns.barplot(x='Demographic Combination in Closed Prompts',
                y='No. Probability Difference',
                hue='Demographic Combination in Open Prompts',
                data=result_df)

    plt.ylabel('No. Probability Difference (%)')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig('./results/bar_plot_in_' + medical_context + ".png")
    plt.clf()
