import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

parser = argparse.ArgumentParser(description='Generate embedding visualizations.')
parser.add_argument('--model', type=str, help="name of the model (xlm-r, x2s-cca or x2s-mse)")
parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
parser.add_argument('--dim', type=int, nargs='?', default=588, help='dimension to analyze, e.g. 588')
parser.add_argument('--job', help="'heatmaps' or 'means' or 'outlier' or 'mean_english'")
parser.add_argument('--language', type=str, nargs='?', default="",
                    help="if only a specific language should be considered")
args = parser.parse_args()

langs = ['ara', 'heb', 'vie', 'ind', 'jav', 'tgl', 'eus', 'mal',
         'tel', 'afr', 'nld', 'deu', 'ell', 'ben', 'hin', 'mar',
         'urd', 'tam', 'fra', 'ita', 'por', 'spa', 'bul', 'rus',
         'jpn', 'kat', 'kor', 'tha', 'swh', 'cmn', 'kaz', 'tur',
         'est', 'fin', 'hun', 'pes']

lang_dict = {'ar': 'ara', 'he': 'heb', 'vi': 'vie', 'in': 'ind',
             'jv': 'jav', 'tl': 'tgl', 'eu': 'eus', 'ml': 'mal',
             'te': 'tel', 'af': 'afr', 'nl': 'nld', 'de': 'deu',
             'el': 'ell', 'bn': 'ben', 'hi': 'hin', 'mr': 'mar', 'ur': 'urd',
             'ta': 'tam', 'fr': 'fra', 'it': 'ita', 'pt': 'por', 'es': 'spa',
             'bg': 'bul', 'ru': 'rus', 'ja': 'jpn', 'ka': 'kat', 'ko': 'kor',
             'th': 'tha', 'sw': 'swh', 'zh': 'cmn', 'kk': 'kaz', 'tr': 'tur',
             'et': 'est', 'fi': 'fin', 'hu': 'hun', 'fa': 'pes'}

lang_dict_3_2 = dict((v, k) for k, v in lang_dict.items())


# produce heatmaps of sentence embeddings per language (not featured in thesis)
def heatmaps(tgt_embs, eng_embs, tgt_lang):
    dir_tgt = f'plots_layer_{args.layer}/{tgt_lang}/{args.model}/{tgt_lang}.png'
    dir_eng = f'plots_layer_{args.layer}/{tgt_lang}/{args.model}/eng.png'
    os.makedirs(os.path.dirname(dir_tgt), exist_ok=True)
    os.makedirs(os.path.dirname(dir_eng), exist_ok=True)

    plt.figure(figsize=(8, 2))
    ax = sns.heatmap(tgt_embs, cmap="YlGnBu", yticklabels=False, xticklabels=100)
    plt.gca().invert_yaxis()
    plt.title(f"{tgt_lang} representations")
    plt.ylabel("# sentence")
    plt.xlabel("dimension")
    plt.savefig(dir_tgt, dpi=300, bbox_inches="tight")
    # plt.show()

    plt.figure(figsize=(8, 2))
    ex = sns.heatmap(eng_embs, cmap="YlGnBu", yticklabels=False, xticklabels=100)
    plt.gca().invert_yaxis()
    plt.title("eng representations")
    plt.ylabel("# sentence")
    plt.xlabel("dimension")
    plt.savefig(dir_eng, dpi=300, bbox_inches="tight")
    # plt.show()
    print(f"Finished {tgt_lang} heatmaps.")


# produce visualizations of language level mean embeddings (for target and english)
def mean_vec(tgt_embs, eng_embs, tgt_lang):
    dir_tgt = f'plots_layer_{args.layer}/{tgt_lang}/{args.model}/mean_{tgt_lang}.png'
    dir_eng = f'plots_layer_{args.layer}/{tgt_lang}/{args.model}/mean_eng.png'
    os.makedirs(os.path.dirname(dir_tgt), exist_ok=True)
    os.makedirs(os.path.dirname(dir_eng), exist_ok=True)

    tgt_mean_vec = torch.mean(tgt_embs, axis=0)
    eng_mean_vec = torch.mean(eng_embs, axis=0)

    plt.figure(figsize=(4, 2))
    plt.plot(tgt_mean_vec)
    plt.title(f"{tgt_lang} mean representation")
    plt.xlabel("dimension")
    plt.savefig(dir_tgt, dpi=300, bbox_inches="tight")
    # plt.show()

    plt.figure(figsize=(4, 2))
    plt.plot(eng_mean_vec)
    plt.title("eng mean representation")
    plt.xlabel("dimension")
    plt.savefig(dir_eng, dpi=300, bbox_inches="tight")
    # plt.show()
    print(f"Finished {tgt_lang} mean vecs.")


# Analyze average english embedding
def mean_english():
    mean_sum_vec = None
    for lang in langs:
        print(f"Processing parallel English of: {lang}")
        eng_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/eng.pt')
        if mean_sum_vec is None:
            mean_sum_vec = torch.mean(eng_emb, axis=0)
        else:
            mean_sum_vec = torch.add(mean_sum_vec, torch.mean(eng_emb, axis=0))

    mean_vec = torch.divide(mean_sum_vec, len(langs))
    mean = torch.mean(mean_vec)
    std_dev = torch.std(mean_vec)

    counter = 0
    outliers = []
    for i in mean_vec:
        if abs(i) > mean + 3 * std_dev:
            outliers.append(counter)
        counter += 1

    print(f"Outlier dimensions: {outliers}")
    for o in outliers:
        print(f"Average value of {o}: {mean_vec[o]}")

    plt.figure(figsize=(6, 1.5))
    plt.plot(mean_vec)
    plt.title(f"average english embedding")
    plt.xlabel("dim")
    plt.savefig("avg_eng_embedding.png", dpi=300, bbox_inches="tight")


# plot sentence level outliers (single values in an outlier dimension)
# can plot either only target lang values or together with english aligned ones
# (comment/uncomment parts accordingly)
def plot_outliers(tgt_embs, eng_embs, tgt_lang):
    dir_tgt = f'dim_plots_layer_{args.layer}/{tgt_lang}/{args.model}/{tgt_lang}_{args.dim}.png'
    dir_tgt_en = f'dim_plots_layer_{args.layer}/{tgt_lang}/{args.model}/eng_{tgt_lang}_{args.dim}.png'
    os.makedirs(os.path.dirname(dir_tgt), exist_ok=True)
    os.makedirs(os.path.dirname(dir_tgt_en), exist_ok=True)

    """dim_values = tgt_embs.T[args.dim][:-1]
    print(f"Max: {torch.amax(dim_values)}")
    print(f"Min: {torch.amin(dim_values)}")"""

    plt.figure(figsize=(5, 1.5))
    plt.plot(tgt_embs.T[args.dim][:-1],
             marker='x', linestyle='None', markersize=2, color='xkcd:soft green', label=tgt_lang)
    # plt.title(f"{lang_dict_3_2[TGT]} dimension {args.dim}")
    # plt.xlabel("#sentence")
    # plt.savefig(dir_tgt, dpi=300, bbox_inches="tight")
    plt.plot(eng_embs.T[args.dim][:-1], marker='x', linestyle='None', markersize=2, color='xkcd:azure', label='en')
    plt.title(f"{lang_dict_3_2[tgt_lang]}-en dimension {args.dim}")
    plt.xlabel("#sentence")
    plt.savefig(dir_tgt_en, dpi=300, bbox_inches="tight")


def main():
    if args.job == "heatmaps":
        print("Generating heatmaps.")
        for lang in langs:
            print(f"Current language: {lang}")
            target_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/{lang}.pt')
            eng_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/eng.pt')
            heatmaps(target_emb, eng_emb, lang)

    elif args.job == "means":
        print("Generating mean representations.")
        for lang in langs:
            print(f"Current language: {lang}")
            target_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/{lang}.pt')
            eng_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/eng.pt')
            mean_vec(target_emb, eng_emb, lang)

    elif args.job == "outlier":
        print("Generating outlier visualization.")
        if args.language != "":
            lang = lang_dict[args.language]
            target_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/{lang}.pt')
            eng_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/eng.pt')
            plot_outliers(target_emb, eng_emb, lang)
        else:
            for lang in langs:
                print(f"Current language: {lang}")
                target_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/{lang}.pt')
                eng_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{lang}/eng.pt')
                plot_outliers(target_emb, eng_emb, lang)

    elif args.job == "mean_english":
        mean_english()
    else:
        exit()


main()
