import argparse
import os

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from post_processing import cluster_based, whitening


def visualise_tsne(embs: np.ndarray, plot_file: str, title: str):
    pca = PCA(n_components=50)
    embs = pca.fit_transform(embs)

    tsne = TSNE(n_components=2, random_state=0)
    embs_2d = tsne.fit_transform(embs)

    clrs = sns.color_palette("pastel", 8)
    fig = plt.figure(figsize=(16, 12))
    # plt.subplot(121)
    plt.title(title)

    tx = embs_2d[:, 0]
    ty = embs_2d[:, 1]
    # when using c=clrs[1], got the following:
    # *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have
    # precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D
    # array with a single row if you intend to specify the same RGB or RGBA value for all points.
    plt.scatter(tx, ty, color=clrs[2])

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def vis_tsne_parallel(embs: np.ndarray, parallel_embs: np.ndarray, plot_file: str, title: str):
    all_embs = np.concatenate((embs, parallel_embs))
    pca = PCA(n_components=50)
    all_embs = pca.fit_transform(all_embs)
    tsne = TSNE(n_components=2, random_state=0)
    embs_2d = tsne.fit_transform(all_embs)

    clrs = sns.color_palette("pastel", 8)
    fig = plt.figure(figsize=(16, 12))
    plt.title(title)

    tx = embs_2d[:, 0]
    ty = embs_2d[:, 1]
    c = [clrs[0] for _ in range(len(embs))] + [clrs[2] for _ in range(len(parallel_embs))]
    plt.scatter(tx, ty, c=c)

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_embs(emb_file, load, do_cbie, do_whiten):
    if load == "torch":
        embs = torch.load(emb_file).numpy()
    elif load == "np":
        embs = np.load(emb_file, allow_pickle=True)
    if do_whiten:
        embs = whitening(embs)
    if do_cbie:  # idk if it's meaningful to do both whitening and cbie, but IF we do both, probably this order
        embs = cluster_based(embs, n_cluster=7, n_pc=12, hidden_size=embs.shape[1])
    return embs


def main(args):
    if not args.plot_file:  # support passing the required attributes instead of a file path directly
        args.plot_file = \
            f'../plots/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/tsne{args.append_file_name}.png'

    if not args.emb_file:  # support passing the required attributes instead of a file path directly
        lang = args.lang_or_track if args.dataset in ['tatoeba', 'wiki'] else 'lng1'
        args.emb_file = \
            f'../embs/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/{lang}{args.append_file_name}.pt'

    # load first set of embeddings
    embs = load_embs(args.emb_file, args.load, args.do_cbie, args.do_whiten)

    if args.parallel_vis:
        if not args.parallel_emb_file:  # support passing the required attributes instead of a file path directly
            lang = args.lang_or_track if args.dataset in ['tatoeba', 'wiki'] else 'lng2'
            args.parallel_emb_file = \
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/{lang}{args.append_file_name}.pt'

        # load parallel set of embeddings
        parallel_embs = load_embs(args.parallel_emb_file, args.load, args.do_cbie, args.do_whiten)

        # visualise
        vis_tsne_parallel(embs, parallel_embs, args.plot_file,
                          f"t-SNE vis of {args.emb_file} and {args.parallel_emb_file}. "
                          f"Whiten: {args.do_whiten} CBIE: {args.do_cbie}")

    else:
        visualise_tsne(embs, args.plot_file, f"t-SNE vis of {args.emb_file}. ")
                                            # f"Whiten: {args.do_whiten} CBIE: {args.do_cbie}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualise embeddings from a file with tSNE')
    parser.add_argument('--model', type=str, help="name of the model to be analyzed")
    parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
    parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki", "sts"],
                        help="use embeddings from this dataset (tatoeba, wiki)")
    parser.add_argument('--append_file_name', type=str, default="", help='to load files à la .._whitened.pt')
    parser.add_argument('--lang_or_track', type=str, default="", help='part of the path')
    parser.add_argument('--emb_file', type=str, help="location of the file in question")
    parser.add_argument('--parallel_emb_file', type=str, required=False, help="location of second emb file")
    parser.add_argument('--parallel_vis', action='store_true', help='if two (parallel) langs should be in same plot')
    parser.add_argument('--plot_file', type=str, help="where to save the plot")
    parser.add_argument('--load', type=str, choices=["torch", "np"], help="library to use for loading [torch, np]")
    parser.add_argument('--do_cbie', action='store_true', help='try doing cbie before')
    parser.add_argument('--do_whiten', action='store_true', help='try doing whitening before')
    args = parser.parse_args()
    main(args)
