from statistics import mean
import pandas as pd
from scipy.stats import ttest_ind

races = ['Black', 'Asian', 'White', 'Hispanic']
genders = ['man', 'woman']
contexts = ['acute_cancer', 'acute_non_cancer', 'chronic_cancer', 'chronic_non_cancer', 'post_op']

# List of dictionaries store the result; conver to dataframe later
ans = []

def create_path(context, bool):
    if bool: 
        return '/Users/zhaoyujian/CMSC848Q_final_project/iterated_results/data_' + context + '_baseline_results.csv'
    else:
        return '/Users/zhaoyujian/CMSC848Q_final_project/iterated_results/data_' + context + '_biased_results.csv'

def ttest(context, race, gender):
    dic = {'context' : context, 'closed_prompt_race': '', 'closed_prompt_gender': '', 'open_prompt_race' : race, 'open_prompt_gender': gender, 't-test-stats': 0, 'p-value': 0, 'group_mean_diff': 0}
    baseline = pd.read_csv(create_path(context, True))
    biased = pd.read_csv(create_path(context, False))

    baseline_select = baseline.loc[baseline['open_prompt_race'] == race]
    baseline_select = baseline_select.loc[baseline_select['open_prompt_gender'] == gender]

    for r in races:
        for g in genders:
            biased_closed = biased.loc[biased['closed_prompt_race'] == r]
            biased_closed = biased_closed.loc[biased_closed['closed_prompt_gender'] == g]

            biased_select = biased_closed.loc[biased_closed['open_prompt_race'] == race]
            biased_select = biased_select.loc[biased_select['open_prompt_gender'] == gender]

            res = ttest_ind(baseline_select['no_prob'], biased_select['no_prob'])
            mean_diff = abs(baseline_select['no_prob'].mean() - biased_select['no_prob'].mean())
            dic['t-test-stats'] = res.statistic
            dic['p-value'] = res.pvalue
            dic['closed_prompt_race'] = r
            dic['closed_prompt_gender'] = g
            dic['group_mean_diff'] = mean_diff
            ans.append(dic)
            dic = {'context' : context,'closed_prompt_race': '', 'closed_prompt_gender': '', 'open_prompt_race' : race, 'open_prompt_gender': gender, 't-test-stats': 0, 'p-value': 0, 'group_mean_diff': 0}

    

# print(ttest('acute_cancer', 'Black', 'man'))
# print(ttest('acute_cancer', 'Black', 'woman'))

for context in contexts:
    for race in races:
        for gender in genders:
            ttest(context, race, gender)

df = pd.DataFrame(ans)

df.to_csv('t-test-result.csv')



