from collections import defaultdict
import torch
from transformers import AutoModel
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Code used to:
# a) Identify magnitude-wise outliers for XLM-R
# b) Identify magnitude-wise outliers per each language (used in chapter 5) + generate visualizations

parser = argparse.ArgumentParser(description='Establish outlier dimensions.')
parser.add_argument('model', type=str, help="name of the model")
parser.add_argument('layer', type=int, help="which model layer the embeddings are from")
parser.add_argument('type', type=str, help="to analyze per-model or per-language ('all' or 'language')")
args = parser.parse_args()

model = AutoModel.from_pretrained(args.device)

# TODO access the weights
if args.type == "all":
    layernorm_weights = None
    layernorm_biases = None
    model.

    if mean_sum_vec is None:
        mean_sum_vec = torch.mean(target_emb, axis=0)
    else:
        mean_sum_vec = torch.add(mean_sum_vec, torch.mean(target_emb, axis=0))

    mean = torch.mean(vector)
    std_dev = torch.std(vector)

    counter = 0
    outliers = []
    for i in vector:
        if abs(i) > mean + 3 * std_dev:
            outliers.append(counter)
        counter += 1

    print(f"Outlier dimensions: {outliers}")
    for o in outliers:
        print(f"Average value of {o}: {vector[o]}")

