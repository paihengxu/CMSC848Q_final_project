# SETTING UP
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch

np.random.seed(42)
import random

random.seed(42)
from utils import *
from const import *
from functools import partial
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.special import softmax

parser = ArgumentParser(description='evaluation')
parser.add_argument('--medical_context_file', type=str,
                    default='data_acute_cancer.csv')
opts = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id, max_length=800)
# model = GPT2LMHeadModel.from_pretrained('gpt2-large', config=config)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

model.cuda(device)
# print(all(p.is_cuda for p in model.parameters()))

# Shuffle the names for each of the ten vignettes
for g in genders:
    for r in races:
        random.shuffle(names[r][g])

# Prompting GPT-3 for each prompt x gender x race combo
# Saving the results as a json

out1 = []
repeat_number = 10

medical_context_file = opts.medical_context_file
vignettes = pd.read_csv(medical_context_file)
data = vignettes[vignettes.Answer == "Yes."]
closed = vignettes[vignettes.Answer == "No."]

# print(closed)
# print(vignettes)

for q in range(10):
    open_prompt_standard = create_open_standard(q)
    print("---------- DOING VIGNETTE #", q, " ----------")

    # Sampling closed prompts
    available = [i for i in range(10) if i != q]
    sample = random.sample(available, 2)
    print(closed)

    closed_promptNo = "Vignette: " + list(closed.Vignette)[0] + "\n" + \
                      "Question: " + list(closed.Question)[0] + "\n" + \
                      "Answer: " + list(closed.Answer)[0] + "\n" + \
                      "Dosage: " + str(closed.Dosage)[0] + "\n" + \
                      "Explanation: " + list(closed.Explanation)[0] + "##\n\n"

    closed_prompt0 = create_closed(sample[0], dose_high)

    # Generate by going through genders and races
    for closed_gender in genders:
        for closed_race in races:
            closed_promptNo = race_gender_closed(closed_promptNo, 'Patient B', closed_gender, closed_race)
            closed_prompt0 = race_gender_closed(closed_prompt0, 'Patient A', closed_gender, closed_race)
            closed_prompt = closed_prompt0 + closed_promptNo

            for g in genders:
                open_prompt_gendered = genderize_open(open_prompt_standard, g)
                for r in races:
                    response = {}  # initialize response for GPT-2
                    open_prompt = race_name_open(open_prompt_gendered, r, q, g)
                    final_prompt = closed_prompt + open_prompt

                    indexed_tokens = tokenizer.encode(final_prompt)
                    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

                    with torch.no_grad():
                        predictions = model(tokens_tensor)
                        results = predictions[0]
                        temp = results[0, -1, :]
                        temp = temp.cpu().numpy()
                        result = softmax(temp)
                        word_1 = tokenizer.encode(' Yes')[0]
                        word_2 = tokenizer.encode(' No')[0]

                    pred_id = torch.argmax(results[0, -1, :]).item()
                    # print("\nPredicted token ID of next word: ")
                    # print(pred_id)

                    pred_word = tokenizer.decode(pred_id)
                    # print("\nPredicted next word for sequence: ")
                    # print(pred_word)

                    response['context'] = medical_context_file.split('.')[0]
                    response['yes_prob'] = result[word_1]
                    response['no_prob'] = result[word_2]
                    response['next_word_pred'] = pred_word
                    response['closed_prompt_race'] = closed_race
                    response['closed_prompt_gender'] = closed_gender
                    response['open_prompt_race'] = r
                    response['open_prompt_gender'] = g
                    response['open_prompt_name'] = names[r][g][q]
                    response['vignette_num'] = q
                    out1.append(response)

# print(out1)
results_data1 = pd.DataFrame(out1)
results_data1.to_csv(medical_context_file.split('.')[0] + '_results.csv')
