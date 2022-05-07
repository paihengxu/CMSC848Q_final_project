import pandas as pd
from scipy.stats import ttest_ind

races = ['Black', 'Asian', 'White', 'Hispanic']
genders = ['man', 'woman']
contexts = ['acute_cancer', 'acute_non_cancer', 'chronic_cancer', 'chronic_non_cancer', 'post_op']

# List of dictionaries store the result; conver to dataframe later
ans = []

def create_path(context, bool):
    if bool: 
        return '/Users/zhaoyujian/CMSC848Q_final_project/results/data_' + context + '_baseline_results.csv'
    else:
        return '/Users/zhaoyujian/CMSC848Q_final_project/results/data_' + context + '_biased_results.csv'

def ttest(context, race, gender):
    dic = {'context' : context, 'race' : race, 'gender': gender, 't-test-stats': 0, 'p-value': 0}
    baseline = pd.read_csv(create_path(context, True))
    biased = pd.read_csv(create_path(context, False))
    baseline_yes = baseline.loc[baseline['open_prompt_race'] == race]
    baseline_yes = baseline_yes.loc[baseline_yes['open_prompt_gender'] == gender]

    biased_yes = biased.loc[biased['open_prompt_race'] == race]
    biased_yes = biased_yes.loc[biased_yes['open_prompt_gender'] == gender]

    res = ttest_ind(baseline_yes['no_prob'], biased_yes['no_prob'])
    dic['t-test-stats'] = res.statistic
    dic['p-value'] = res.pvalue
    ans.append(dic)

# print(ttest('acute_cancer', 'Black', 'man'))
# print(ttest('acute_cancer', 'Black', 'woman'))

for context in contexts:
    for race in races:
        for gender in genders:
            ttest(context, race, gender)

df = pd.DataFrame(ans)

df.to_csv('t-test-result.csv')

