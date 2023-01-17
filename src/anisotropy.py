# Adjusted original code from https://github.com/wtimkey/rogue-dimensions/blob/main/replication.ipynb
import numpy as np
import torch
import random
import argparse

from constants import langs_tatoeba, langs_wiki, sts_tracks


def cos_contrib(emb1, emb2):
    numerator_terms = emb1 * emb2
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return np.array(numerator_terms / denom)


def remove_dims(dim, emb):
    emb_transposed = emb.T.clone().detach()
    emb_transposed[dim] = 0
    # transpose back into original format
    return emb_transposed.T


def main(args):
    if args.dataset == "tatoeba":
        langs = langs_tatoeba
    elif args.dataset == "wiki":
        langs = langs_wiki
    elif args.dataset == "sts":
        langs = sts_tracks
    else:
        raise ValueError("unknown dataset argument")

    avg_anisotropy = 0
    mean_contribs = []
    for lang in langs:
        # print(f"Current language: {lang}")

        layer_cosine_contribs = []
        if args.dataset == 'tatoeba' or args.dataset == 'sts':
            if args.dataset == 'tatoeba':
                target_embs = torch.load(
                    f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}{args.append_file_name}.pt')
                eng_embs = torch.load(
                    f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng{args.append_file_name}.pt')
            elif args.dataset == 'sts':
                target_embs = torch.load(
                    f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/lng1{args.append_file_name}.pt')
                eng_embs = torch.load(
                    f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/lng2{args.append_file_name}.pt')

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
            target_embs = torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}{args.append_file_name}.pt')

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

        layer_cosine_contribs = np.stack(layer_cosine_contribs)
        layer_cosine_contribs_mean = layer_cosine_contribs.mean(axis=0)

        aniso = layer_cosine_contribs_mean.sum()
        avg_anisotropy += aniso
        mean_contribs.append(layer_cosine_contribs_mean)

        top_dims = np.argsort(layer_cosine_contribs_mean)[-10:]
        top_dims = np.flip(top_dims)
        top = layer_cosine_contribs_mean[top_dims[0]] / aniso
        second = layer_cosine_contribs_mean[top_dims[1]] / aniso
        third = layer_cosine_contribs_mean[top_dims[2]] / aniso
        fourth = layer_cosine_contribs_mean[top_dims[3]] / aniso
        fifth = layer_cosine_contribs_mean[top_dims[4]] / aniso
        six = layer_cosine_contribs_mean[top_dims[5]] / aniso
        seven = layer_cosine_contribs_mean[top_dims[6]] / aniso
        eight = layer_cosine_contribs_mean[top_dims[7]] / aniso
        nine = layer_cosine_contribs_mean[top_dims[8]] / aniso
        ten = layer_cosine_contribs_mean[top_dims[9]] / aniso

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
    avg_anisotropy = avg_anisotropy / len(langs)
    print(f"Average Anisotropy: {avg_anisotropy}")

    mean_contribs = np.stack(mean_contribs).mean(axis=0)
    top_dims = np.argsort(mean_contribs)[-10:]
    top_dims = np.flip(top_dims)
    print(f"Top 10 dims: {top_dims}")
    print("Mean cosine contributions:")
    for i in range(10):
        d = top_dims[i]
        print(d, mean_contribs[d] / avg_anisotropy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze anisotropy behaviour.')
    parser.add_argument('--model', type=str, help="name of the model to be analyzed")
    parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
    parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki"],
                        help="use embeddings from this dataset (tatoeba, wiki)")
    parser.add_argument('--append_file_name', type=str, default="", help='to load files à la .._whitened.pt')
    parser.add_argument('--dim', type=int, nargs='*', default=-1, help="which dimension to zero out if any")

    args = parser.parse_args()
    main(args)
