import torch
from torch import nn
import argparse
from collections import defaultdict
import operator

# Code for checking parallel data cosines (and establishing similarity-harming group)

parser = argparse.ArgumentParser(description='')
parser.add_argument('model', type=str, help="name of the model (xlm-r, x2s-cca or x2s-mse)")
parser.add_argument('hidden_size', type=int, default=768, help="model hidden size")
parser.add_argument('layer', type=int, help="which model layer the embeddings are from")
parser.add_argument('dim', type=int, nargs='?', default=-1, help='dimension to analyze, e.g. 588')
parser.add_argument('language', type=str, nargs='?', default="",
                    help="if only a specific language should be considered")
args = parser.parse_args()

langs = ['ara', 'heb', 'vie', 'ind', 'jav', 'tgl', 'eus', 'mal',
         'tel', 'afr', 'nld', 'deu', 'ell', 'ben', 'hin', 'mar',
         'urd', 'tam', 'fra', 'ita', 'por', 'spa', 'bul', 'rus',
         'jpn', 'kat', 'kor', 'tha', 'swh', 'cmn', 'kaz', 'tur',
         'est', 'fin', 'hun', 'pes']


def calc_cosine(tgt, eng):
    cos = nn.CosineSimilarity()
    return torch.mean(cos(tgt, eng))


def remove_dims(tgt, eng, dim):
    tgt_transposed, eng_transposed = tgt.T.clone().detach(), eng.T.clone().detach()
    tgt_transposed[dim] = 0
    eng_transposed[dim] = 0
    # transpose back into original format
    return tgt_transposed.T, eng_transposed.T


def check_parallel_cosine():
    if args.language != "":
        languages = [args.language]
    else:
        languages = langs

    differences = defaultdict(lambda: 0)
    for TGT in languages:
        print(f"Language: {TGT}")
        target_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{TGT}/{TGT}.pt')
        eng_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{TGT}/eng.pt')

        original_cos = calc_cosine(target_emb, eng_emb)
        print(f"Parallel Data Cosine Sim: {original_cos}")

        for dim in range(0, args.hidden_size):
            new_target_emb, new_eng_emb = remove_dims(target_emb, eng_emb, dim)
            removed_dim_cos = calc_cosine(new_target_emb, new_eng_emb)
            diff = removed_dim_cos - original_cos
            differences[dim] += diff.item()
    differences = {k: v / len(langs) for k, v in differences.items()}
    sorted_d = dict(sorted(differences.items(), key=operator.itemgetter(1), reverse=True))
    print(sorted_d)


def main():
    check_parallel_cosine()


main()
