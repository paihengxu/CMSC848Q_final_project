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

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', max_length=1024)

# Initializing a GPT2 configuration
configuration = GPT2Config()

# Initializing a model from the configuration
model = GPT2LMHeadModel(configuration)

# Accessing the model configuration
configuration = model.config
# print(configuration)

vignettes = pd.read_csv("data_acute_cancer.csv")
data = vignettes[vignettes.Answer == "Yes."]
closed = vignettes[vignettes.Answer == "No."]

print(closed)
print(vignettes)

closed_promptNo = "Vignette: " + list(closed.Vignette)[0] + "\n" + \
                  "Question: " + list(closed.Question)[0] + "\n" + \
                  "Answer: " + list(closed.Answer)[0] + "\n" + \
                  "Dosage: " + str(closed.Dosage)[0] + "\n" + \
                  "Explanation: " + list(closed.Explanation)[0] + "##\n\n"
print(closed_promptNo)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.cuda(device)
print(all(p.is_cuda for p in model.parameters()))

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
    closed_prompt2 = standardize_closed(closed_prompt2, 'Patient C')
    closed_prompt = closed_prompt0 + closed_promptNo  # + closed_prompt2

    # Generate by going through genders and races
    for g in genders:
        open_prompt_gendered = genderize_open(open_prompt_standard, g)
        for r in races:
            open_prompt = race_name_open(open_prompt_gendered, r, q)
            final_prompt = closed_prompt + open_prompt

            '''
            # for getting loss
            final_prompt = closed_prompt + open_prompt + '{}'
            print(score(final_prompt.format('no')))
            break
            '''

            encoded_input = tokenizer.encode(final_prompt, return_tensors='pt', max_length=1024, truncation=True)
            print(type(encoded_input))
            encoded_input = encoded_input.to(device)
            print(encoded_input.is_cuda)
            outputs = model.generate(encoded_input, return_dict_in_generate=True, output_scores=True)
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            print(response)

            '''
            # get probability
            # This gets probability of predicted word, but when it's not "no", we cannot know the probability of "no"
            # follow code from https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
            gen_sequences = outputs.sequences[:, encoded_input.shape[-1]:]
            print(gen_sequences.shape)
            print(outputs)
            next_token_logits = outputs[0][:, -1, :]
            print(next_token_logits)
            probs = torch.stack(outputs.scores, dim=1).softmax(-1)
            print(probs)
            print(probs.shape)

            gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            print(gen_probs)
            '''
            break

            # EXAMPLE WITH GPT-3 OPEN AI API / REPLACE WITH YOUR OWN EXPERIMENT
            # openai.api_key = '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'
            # response = openai.Completion.create(engine="davinci", prompt=final_prompt, max_tokens=max_tokens, temperature=temp, n=1, logprobs=logp, stop=stop)
            response['closed_prompt'] = closed_prompt
            response['open_prompt'] = open_prompt
            response['prompt_num'] = q
            response['race'] = r
            response['gender'] = g
            response['name'] = names[r][g][q]
            out1.append(response)
        break
    break

# test_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# test_model = GPT2LMHeadModel.from_pretrained('gpt2')
#
# inputs = test_tokenizer("Hello, my dog is cute and ", return_tensors="pt")
# input_ids = inputs.input_ids
# generation_output = test_model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
# print(generation_output)
#
# # follow code from https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
# gen_sequences = generation_output.sequences[:, input_ids.shape[-1]:]
# print(gen_sequences.shape)
# probs = torch.stack(generation_output.scores, dim=1).softmax(-1)
# print(probs)
# print(probs.shape)
#
# gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
# print(gen_probs)
#
# outputs = test_model(input_ids)
# next_token_logits = outputs[0][:, -1, :]
# print(outputs)
# print(next_token_logits)
#
# results_data1 = pd.DataFrame(columns=['context', 'vignette_idx', 'name', 'gender', 'race', 'prob_gpt3_yes',
#                                       'prob_gpt3_no', 'prob_gpt3_high', 'prob_gpt3_low', 'gpt3_answer',
#                                       'gpt3_dosage', 'gpt3_explanation', 'gpt3_full', 'closed_prompts', 'open_prompts'])
#
# for generated in out1:
#     # Getting the probs for the first and second generated tokens (skipping the punctuation)
#     logprobs_yn = generated['choices'][0]['logprobs']['top_logprobs'][0]
#     logprobs_dosage = generated['choices'][0]['logprobs']['top_logprobs'][7]
#
#     # Finding the probs for our chosen words
#     options_yn_prob = {option: np.exp(logprobs_yn.get(' ' + str(option), -10000)) for option in OPTIONS_YESNO}
#     options_dosage_prob = {option: np.exp(logprobs_dosage.get(' ' + str(option), -10000)) for option in OPTIONS_DOSAGE}
#
#     prob_yes = options_yn_prob['Yes']
#     prob_no = options_yn_prob['No']
#
#     prob_high = options_dosage_prob['High']
#     prob_low = options_dosage_prob['Low']
#
#     # GPT-3 Output
#     gen_text = generated['choices'][0]['text']
#     prompt = generated['open_prompt']
#     closed = generated['closed_prompt']
#     q = generated['prompt_num']
#     r = generated['race']
#     g = generated['gender']
#     name = generated['name']
#
#     split_answer = gen_text.split(sep="\n")
#     answer = split_answer[0]
#     dosage = split_answer[1]
#     explanation = split_answer[2]
#
#     new_idx = ['Acute Cancer Pain', q, name, g, r, prob_yes,
#                prob_no, prob_high, prob_low, answer, dosage, explanation, gen_text, closed, prompt]
#     results_data1 = results_data1.append(dict(zip(results_data1.columns, new_idx)), ignore_index=True)
#
# print(results_data1)
