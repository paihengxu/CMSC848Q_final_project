#!/usr/bin/env bash

for fn in "data_acute_cancer.csv" "data_acute_non_cancer.csv" "data_chronic_cancer.csv" "data_chronic_non_cancer.csv" "data_post_op.csv";
do
    for setting in "biased" "baseline";
    do
        sbatch --job-name ${fn}_${setting} --output iterated_results/logs/${fn}_${setting}.log run_paiheng.sh ${fn} ${setting}
    done
done