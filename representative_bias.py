import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from utils import *


def get_representative_measure(df):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # generate standard prompt without special tokens
    df['prompt'] = df.apply(my_create_closed, axis=1)
    df['prompt'] = df['prompt'].apply(my_standardize_closed)

    # generate embedding for each prompt
    df['prompt_embedding'] = df['prompt'].apply(model.encode)

    # get pairwise mean embedding
    mean_embed = np.mean(df['prompt_embedding'].values, axis=0)

    def get_dist_mean_embed(embed):
        return 1 - cosine(embed, mean_embed)

    # calculate the similarity to the mean embedding. The larger, the more similar
    df['representative_sim_prompt'] = df['prompt_embedding'].apply(get_dist_mean_embed)

    # also compute the cluster for vignettes
    df['vignette_embedding'] = df['Vignette'].apply(model.encode)
    mean_embed = np.mean(df['vignette_embedding'].values, axis=0)
    df['representative_sim_vignette'] = df['vignette_embedding'].apply(get_dist_mean_embed)
    return df


if __name__ == '__main__':
    for fn in ["data_acute_cancer.csv", "data_acute_non_cancer.csv", "data_chronic_cancer.csv",
               "data_chronic_non_cancer.csv", "data_post_op.csv"]:
        print(fn)
        vignettes = pd.read_csv(fn)
        vignettes_w_prompt = get_representative_measure(vignettes)
        new_fn = fn.split('.')[0]
        vignettes_w_prompt.to_csv(f"processed_data/{new_fn}_w_representation.csv")
