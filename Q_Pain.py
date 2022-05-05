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
from transformers import GPT2Config
from scipy.special import softmax

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id, max_length=700)
# model = GPT2LMHeadModel.from_pretrained('gpt2-large', config=config)

vignettes = pd.read_csv("data_acute_cancer.csv")
data = vignettes[vignettes.Answer == "Yes."]
closed = vignettes[vignettes.Answer == "No."]

# print(closed)
# print(vignettes)

closed_promptNo = "Vignette: " + list(closed.Vignette)[0] + "\n" + \
                  "Question: " + list(closed.Question)[0] + "\n" + \
                  "Answer: " + list(closed.Answer)[0] + "\n" + \
                  "Dosage: " + str(closed.Dosage)[0] + "\n" + \
                  "Explanation: " + list(closed.Explanation)[0] + "##\n\n"

# closed_promptNo = asia_man_closed(closed_promptNo, 'Patient B')
# print(closed_promptNo)

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
# Shuffle the names for each of the ten vignettes
for g in genders:
    for r in races:
        random.shuffle(names[r][g])

# Prompting GPT-3 for each prompt x gender x race combo
# Saving the results as a json

out1 = []
for q in range(10):
    open_prompt_standard = create_open_standard(q)
    print("---------- DOING VIGNETTE #", q, " ----------")

    # Sampling closed prompts
    available = [i for i in range(10) if i != q]
    sample = random.sample(available, 2)
    closed_prompt0 = create_closed(sample[0], dose_high)
    closed_prompt2 = create_closed(sample[1], dose_low)

    # Standardizing closed prompts
    closed_prompt0 = standardize_closed(closed_prompt0, 'Patient A')
    closed_prompt = closed_prompt0 + closed_promptNo  # + closed_prompt2

    # Generate by going through genders and races
    for g in genders:
        open_prompt_gendered = genderize_open(open_prompt_standard, g)
        for r in races:
            response = {}  # initialize response for GPT-2
            open_prompt = race_name_open(open_prompt_gendered, r, q, g)
            final_prompt = closed_prompt + open_prompt

            '''
            # for getting loss
            final_prompt = closed_prompt + open_prompt + '{}'
            print(score(final_prompt.format('no')))
            break
            '''
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
                print(result[word_1])
                print(result[word_2])
                # print(result)
                # print(sum(result), len(result))

            pred_id = torch.argmax(results[0, -1, :]).item()
            # print("\nPredicted token ID of next word: ")
            # print(pred_id)

            pred_word = tokenizer.decode(pred_id)
            print("\nPredicted next word for sequence: ")
            print(pred_word)

            encoded_input = tokenizer.encode(final_prompt, return_tensors='pt', max_length=1024, truncation=True)
            # print(type(encoded_input))
            encoded_input = encoded_input.to(device)
            # print(encoded_input.is_cuda)
            outputs = model.generate(encoded_input, return_dict_in_generate=True, output_scores=True)
            response_str = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            # print(response_str.split()[-1])

            response['yes_prob'] = result[word_1]
            response['no_prob'] = result[word_2]
            response['next_word_pred'] = pred_word
            response['next_word_gen'] = response_str.split()[-1]
            response['closed_prompt'] = closed_prompt
            response['open_prompt'] = open_prompt
            response['prompt_num'] = q
            response['race'] = r
            response['gender'] = g
            response['name'] = names[r][g][q]
            out1.append(response)
#         break
#     break
