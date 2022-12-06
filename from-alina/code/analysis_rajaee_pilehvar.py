""" Adapted from github.com/Sara-Rajaee/Multilingual-Isotropy"""
import argparse
from typing import List

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel, AutoModel, AutoConfig
import tensorflow as tf
import pickle
import scipy as sc
import math as mt
from sklearn.metrics.pairwise import cosine_similarity
import random
from random import randint
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


# Loading mBERT
# todo i want to use pytorch and automodel, would that somehow change the results??
# casing = "bert-base-multilingual-uncased"
# tokenizer_mBERT = BertTokenizer.from_pretrained(casing, do_lower_case=True, add_special_tokens=True)
# config = BertConfig(casing, output_hidden_states=True)
# mBERT = TFBertModel.from_pretrained(casing)
# mBERT.trainable = False


def load_model(model_name="bert-base-multilingual-uncased"):
    config = AutoConfig(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True,
                                              add_special_tokens=True)  # todo do_lower_case=False for other models
    model = AutoModel.from_pretrained(model_name)  # TFAutoModel.from_pretrained(model_name)
    model.trainable = False
    return config, tokenizer, model


def text2rep(model, tokenizer, df, dimension=768):
    representation = []
    for i in range(len(df)):

        inputs = tokenizer.encode(df['Sentence'].iloc[i], add_special_tokens=False)
        inputs = np.asarray(inputs, dtype='int32').reshape((1, -1))
        output = model(inputs)[0]
        for j in range(output.shape[1]):
            representation.append(output[0][j])

    return representation


def isotropy(representations):
    """Calculating isotropy of embedding space based on I_PC
           arg:
              representations (n_samples, n_dimensions)
            """

    eig_values, eig_vectors = np.linalg.eig(np.matmul(np.transpose(representations), representations))
    max_f = -mt.inf
    min_f = mt.inf

    for i in range(eig_vectors.shape[1]):
        f = np.matmul(representations, np.expand_dims(eig_vectors[:, i], 1))
        f = np.sum(np.exp(f))

        min_f = min(min_f, f)
        max_f = max(max_f, f)

    isotropy = min_f / max_f

    return isotropy


def cosine_contribution(representation, n):
    contribution = []
    num = 1000
    for i in range(num):
        n1 = random.randint(0, len(representation) - 1)
        n2 = random.randint(0, len(representation) - 1)
        if n1 != n2:
            contribution.append(np.multiply(representation[n1], representation[n2]) /
                                (np.linalg.norm(representation[n1]) * np.linalg.norm(representation[n2])))

    sim = []
    for i in range(num):
        n1 = random.randint(0, len(representation) - 1)
        n2 = random.randint(0, len(representation) - 1)
        if n1 != n2:
            sim.append(cosine_similarity(
                np.reshape(representation[n1], (1, -1)), np.reshape(representation[n2], (1, -1)))[0][0])

    ary = np.mean(contribution, axis=0) / np.mean(sim)
    return ary[np.argsort(ary)[-n:]]


def extract_reps_per_lang(model, tokenizer, df, lang, base_dir, model_name):
    rep = text2rep(model, tokenizer, df)
    rd = random.sample(range(0, len(rep)), 10000)
    selected = [rep[idx] for idx in rd]  # todo isn't that the same as = rep[rd]?
    np.save(f'{base_dir}/{model_name}/selected_{lang}.npy', np.asarray(selected))


def get_contributions(langs: List[str], base_dir, model_name):
    model_contribution = []
    for lang in langs:
        selected = np.load(f'{base_dir}/{model_name}/selected_{lang}.npy')
        model_contribution.append(cosine_contribution(selected, 3))
        print(isotropy(selected))
    model_contribution = np.asarray(model_contribution)

    t_file = open(f'{model_name}-contribution.txt', "w")
    for row in model_contribution:
        np.savetxt(t_file, row)
    t_file.close()


# Visualization
def visualise(selected, lang, fig_dir, hidden_size=768):
    plt.rcParams["figure.figsize"] = (30, 3)
    x = np.arange(hidden_size)
    st = np.std(selected) * 3
    m = np.mean(selected)

    figure(figsize=(23, 3), dpi=80)
    fig = plt.bar(x, np.mean(selected, axis=0), color='#073b4c', width=2)
    plt.xlim([0, hidden_size])
    plt.ylim([-2, 2])
    plt.xticks(fontsize=30)
    clrs = sns.color_palette("pastel", 8)
    plt.yticks(fontsize=30)
    plt.yticks(np.arange(-2, 2.1, 2))
    plt.fill_between(x, m - st, m + st, alpha=0.5, facecolor=clrs[7])

    plt.title(f"{lang}")
    plt.savefig(fig_dir, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Isotropy analysis, ref. Rajaee and Pilehvar 2022')
    parser.add_argument('--model', type=str, help="model name")
    parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
    parser.add_argument('--base_dir', type=str, help="where to save/load representations for the analysis")
    # parser.add_argument('--overwrite', type=bool, default=False,
    #                    help="whether to redo extracting embeddings for which a file already exists")


    # Loading Wikipedia datasets
    df_su = pd.read_csv('/content/Sundanese.csv', sep=',')
    df_sw = pd.read_csv('/content/Swahili.csv', sep=',')
    df_en = pd.read_csv('/content/English.csv', sep=',')
    df_es = pd.read_csv('/content/Spanish.csv', sep=',')
    df_ar = pd.read_csv('/content/Arabic.csv', sep=',')
    df_tr = pd.read_csv('/content/Turkish.csv', sep=',')

    langs = ['su', 'sw', 'en', 'es', 'ar', 'tr']
    dfs = [df_su, df_sw, df_en, df_es, df_ar, df_tr]


