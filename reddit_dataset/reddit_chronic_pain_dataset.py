# -*- coding: utf-8 -*-
import json
import sys

from pyspark import SparkConf, SparkContext

sc_name = "Reddit Dataset"
sc = SparkContext(
    conf=SparkConf().set("spark.files.ignoreCorruptFiles", "true").set("spark.default.parallelism", 100).setAppName(
        sc_name))


def main():
    output = "/user/tonyzhou/reddit/"
    reddit_dir = "/var/reddit/"
    output_folder = "chronic_pain_dataset"
    input_file = sys.argv[1]
    files = sc.textFile(reddit_dir + input_file)

    text = files.map(json.loads).filter(lambda t: t["subreddit"] == "ChronicPain" and "selftext" in t).map(
        json.dumps).repartition(1).saveAsTextFile(output + output_folder)
    sc.stop()


if __name__ == '__main__':
    main()
