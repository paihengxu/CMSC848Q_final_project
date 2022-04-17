from argparse import ArgumentParser
import pandas as pd
import numpy as np

np.random.seed(42)
import random

random.seed(42)
from functools import partial
import json
from const import *

# FUNCTIONS TO SET UP CLOSED PROMPTS

def create_closed(i, dose):
    p = "Vignette: " + data.Vignette[i] + "\n" + \
        "Question: " + data.Question[i] + "\n" + \
        "Answer: " + data.Answer[i] + "\n" + \
        dose + "\n" + \
        "Explanation: " + data.Explanation[i] + "##\n\n"
    return p


def my_create_closed(data, dose=dose_high):
    p = "Vignette: " + data['Vignette'] + "\n" + \
        "Question: " + data['Question'] + "\n" + \
        "Answer: " + data['Answer'] + "\n" + \
        dose + "\n" + \
        "Explanation: " + data['Explanation'] + "##\n\n"
    return p


def standardize_closed(p, patient):
    p = p.replace('[gender] ', '')
    p = p.replace('[race] ', '')
    p = p.replace('[possessive]', 'their')
    p = p.replace('[subject]', patient)
    p = p.replace('Patient D', patient)
    return p


def my_standardize_closed(p):
    p = p.replace('[gender] ', '')
    p = p.replace('[race] ', '')
    p = p.replace('[possessive]', 'their')
    p = p.replace('[subject]', 'Patient D')
    p = p.replace('Patient B', 'Patient D')
    return p


# FUNCTIONS TO SET UP OPEN PROMPTS

def create_open_standard(q):
    p = "Vignette: " + data.Vignette[q] + "\n" + \
        "Question: " + data.Question[q] + "\n" + \
        "Answer:"
    return p


def genderize_open(p, g):
    p = p.replace('[gender]', g)
    p = p.replace('[possessive]', pronouns['possessive'][g])
    p = p.replace('[subject]', pronouns['subject'][g])
    return p


def race_name_open(p, r, q):
    p = p.replace('[race]', r)
    p = p.replace('Patient D', names[r][g][q])
    return p
