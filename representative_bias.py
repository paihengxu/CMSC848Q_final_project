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
    df['representative_sim'] = df['prompt_embedding'].apply(get_dist_mean_embed)

    return df


if __name__ == '__main__':
    vignettes = pd.read_csv("data_acute_cancer.csv")
    vignettes_w_prompt = get_representative_measure()
    # data = vignettes[vignettes.Answer == "Yes."]
    # closed = vignettes[vignettes.Answer == "No."]