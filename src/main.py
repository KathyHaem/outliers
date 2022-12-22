import argparse
import json

import numpy as np
import torch

import anisotropy
import extract_sent_embeddings
import find_outliers
import plots
from constants import langs_tatoeba_2
from scripts.third_party.evaluate import evaluate
from scripts.third_party.evaluate_retrieval import predict_tatoeba


def main():
    models = ["xlm-roberta-base", "bert-base-multilingual-cased",
              "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"]  # (which) others?

    for model in models:
        args = argparse.Namespace(model=model, layer=7, device='cpu', dataset='tatoeba', batch_size=64,
                                  save_whitened=True, save_cbie=True)
        extract_analyse(args)

        # checking tatoeba performances
        args = argparse.Namespace(dist='cosine', embed_size=768, tgt_language='en')
        for lang in langs_tatoeba_2:
            args.src_language = lang
            tatoeba(args, 'unmod', lang, model)
            tatoeba(args, 'whitened', lang, model)
            tatoeba(args, 'cbie', lang, model)

        # doing similar things for wiki dataset
        args = argparse.Namespace(model=model, layer=7, device='cpu', dataset='wiki', batch_size=64,
                                  save_whitened=True, save_cbie=True)

        extract_analyse(args)


def extract_analyse(args):
    extract_sent_embeddings.main(args)
    # outlier/anisotropy analysis on unmodified embeds
    args.append_file_name = ""
    args.stdevs = 3
    args.type = 'all'  # should i be doing 'language' (as well)?
    find_outliers.main(args)
    args.dim = None
    anisotropy.main(args)
    args.job = 'means'
    args.language = ''
    plots.main(args)
    # cbie version
    args.append_file_name = "_cbie"
    find_outliers.main(args)
    anisotropy.main(args)
    plots.main(args)
    # whitening version
    args.append_file_name = "_whitened"
    find_outliers.main(args)
    anisotropy.main(args)
    plots.main(args)


def tatoeba(args, post_proc, lang, model):
    args.predict_dir = f'../predictions/{model}/{post_proc}/tatoeba/'
    append = '' if post_proc == 'unmod' else f'_{post_proc}'
    src = torch.load(f'{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}{append}.pt').numpy().astype(np.float32)
    tgt = torch.load(f'{args.dataset}/{args.model}/{args.layer}/{lang}/eng{append}.pt').numpy().astype(np.float32)
    predict_tatoeba(args, src, tgt)
    overall_scores, detailed_scores = evaluate(
        prediction_folder=f'../predictions/{model}/{post_proc}/', label_folder='../data/labels/'
    )
    overall_scores.update(detailed_scores)
    print(json.dumps(overall_scores))


if __name__ == "__main__":
    main()
