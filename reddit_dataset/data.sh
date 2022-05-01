export PYSPARK_PYTHON=python2.7
export PYSPARK_DRIVER_PYTHON=python2.7

spark-submit --master yarn --queue umsi-qmei \
--num-executors 40 --executor-cores 5 --executor-memory 10G \
reddit_chronic_pain_dataset.py RS_2019-12