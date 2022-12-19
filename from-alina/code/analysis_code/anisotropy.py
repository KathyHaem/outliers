# Adjusted original code from https://github.com/wtimkey/rogue-dimensions/blob/main/replication.ipynb
import numpy as np
import torch
import random
import argparse

parser = argparse.ArgumentParser(description='Analyze anisotropy behaviour.')
parser.add_argument('--model', type=str, help="name of the model to be analyzed")
parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki"],
                    help="use embeddings from this dataset (tatoeba, wiki)")
parser.add_argument('--dim', type=int, nargs='*', default=-1, help="which dimension to zero out if any")

args = parser.parse_args()

langs_tatoeba = ['ara', 'heb', 'vie', 'ind', 'jav', 'tgl', 'eus', 'mal',
                 'tel', 'afr', 'nld', 'deu', 'ell', 'ben', 'hin', 'mar',
                 'urd', 'tam', 'fra', 'ita', 'por', 'spa', 'bul', 'rus',
                 'jpn', 'kat', 'kor', 'tha', 'swh', 'cmn', 'kaz', 'tur',
                 'est', 'fin', 'hun', 'pes']

langs_wiki = ['ara', 'eng', 'spa', 'sun', 'swh', 'tur']

lang_dict = {'ar': 'ara', 'he': 'heb', 'vi': 'vie', 'in': 'ind',
             'jv': 'jav', 'tl': 'tgl', 'eu': 'eus', 'ml': 'mal',
             'te': 'tel', 'af': 'afr', 'nl': 'nld', 'de': 'deu', 'en': 'eng',
             'el': 'ell', 'bn': 'ben', 'hi': 'hin', 'mr': 'mar', 'ur': 'urd',
             'ta': 'tam', 'fr': 'fra', 'it': 'ita', 'pt': 'por', 'es': 'spa',
             'bg': 'bul', 'ru': 'rus', 'ja': 'jpn', 'ka': 'kat', 'ko': 'kor',
             'th': 'tha', 'sw': 'swh', 'zh': 'cmn', 'kk': 'kaz', 'tr': 'tur',
             'et': 'est', 'fi': 'fin', 'hu': 'hun', 'fa': 'pes', 'su': 'sun'}

lang_dict_3_2 = dict((v, k) for k, v in lang_dict.items())

if args.dataset == "tatoeba":
    langs = langs_tatoeba
elif args.dataset == "wiki":
    langs = langs_wiki
else:
    raise ValueError("unknown dataset argument")


def cos_contrib(emb1, emb2):
    numerator_terms = emb1 * emb2
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return np.array(numerator_terms / denom)


def remove_dims(dim, emb):
    emb_transposed = emb.T.clone().detach()
    emb_transposed[dim] = 0
    # transpose back into original format
    return emb_transposed.T


def main():
    avg_anisotropy = 0
    for lang in langs:
        print(f"Current language: {lang}")
        cos_contribs_by_layer = []
        layer_cosine_contribs = []
        if args.dataset == 'tatoeba':
            target_embs = torch.load(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')
            eng_embs = torch.load(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng.pt')

            # if some dimension should be zeroed out first
            if args.dim != -1:
                target_embs = remove_dims(args.dim, target_embs)
                eng_embs = remove_dims(args.dim, eng_embs)
            num_sents = target_embs.shape[0]
            # randomly sample embedding pairs
            random_pairs = [random.sample(range(num_sents), 2) for _ in range(10000)]

            for pair in random_pairs:
                emb1, emb2 = target_embs[pair[0]], eng_embs[pair[1]]
                layer_cosine_contribs.append(cos_contrib(emb1, emb2))

        elif args.dataset == 'wiki':
            target_embs = torch.load(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')

            # if some dimension should be zeroed out first
            if args.dim != -1:
                target_embs = remove_dims(args.dim, target_embs)

            # randomly sample embedding pairs
            num_sents = target_embs.shape[0]
            random_pairs = [random.sample(range(num_sents), 2) for _ in range(10000)]

            for pair in random_pairs:
                if pair[0] == pair[1]:
                    continue
                emb1, emb2 = target_embs[pair[0]], target_embs[pair[1]]
                layer_cosine_contribs.append(cos_contrib(emb1, emb2))

        layer_cosine_contribs = np.array(layer_cosine_contribs)
        # layer_cosine_sims = layer_cosine_contribs.sum(axis=1)
        layer_cosine_contribs_mean = layer_cosine_contribs.mean(axis=0)

        cos_contribs_by_layer.append(layer_cosine_contribs_mean)
        cos_contribs_by_layer = np.array(cos_contribs_by_layer)

        aniso = cos_contribs_by_layer.sum(axis=1)
        avg_anisotropy += aniso

        top_dims = np.argsort(layer_cosine_contribs_mean)[-10:]
        top_dims = np.flip(top_dims)
        top = cos_contribs_by_layer[0, top_dims[0]] / aniso[0]
        second = cos_contribs_by_layer[0, top_dims[1]] / aniso[0]
        third = cos_contribs_by_layer[0, top_dims[2]] / aniso[0]
        fourth = cos_contribs_by_layer[0, top_dims[3]] / aniso[0]
        fifth = cos_contribs_by_layer[0, top_dims[4]] / aniso[0]
        six = cos_contribs_by_layer[0, top_dims[5]] / aniso[0]
        seven = cos_contribs_by_layer[0, top_dims[6]] / aniso[0]
        eight = cos_contribs_by_layer[0, top_dims[7]] / aniso[0]
        nine = cos_contribs_by_layer[0, top_dims[8]] / aniso[0]
        ten = cos_contribs_by_layer[0, top_dims[9]] / aniso[0]

        print(f"### {lang} ###")
        print(f"Top 10 dims: {top_dims}")
        print(f"Estimated anisotropy: {aniso}")
        print("Contributions to expected cosine sim between random embeddings:")
        print(top_dims[0], top)
        print(top_dims[1], second)
        print(top_dims[2], third)
        print(top_dims[3], fourth)
        print(top_dims[4], fifth)
        print(top_dims[5], six)
        print(top_dims[6], seven)
        print(top_dims[7], eight)
        print(top_dims[8], nine)
        print(top_dims[9], ten)
    print()
    print(f"Average Anisotropy: {avg_anisotropy / len(langs)}")


main()
