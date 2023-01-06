import argparse
import json

import numpy as np
import torch

import anisotropy
import extract_sent_embeddings
import find_outliers
import plots
from constants import langs_tatoeba_2, lang_dict, sts_gold_files
from scripts.third_party.evaluate import evaluate
from scripts.third_party.evaluate_retrieval import predict_tatoeba
from sklearn.metrics.pairwise import paired_cosine_distances


def main():
    models = ["xlm-roberta-base", "bert-base-multilingual-cased",
              "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"]  # (which) others?

    for model in models:
        args = argparse.Namespace(model=model, layer=7, device='0', dataset='tatoeba', tatoeba_use_task_order=True,
                                  batch_size=64, save_whitened=True, save_cbie=True)
        extract_analyse(args)

        # checking tatoeba performances
        args = argparse.Namespace(dist='cosine', layer=7, embed_size=768, tgt_language='en')
        tatoeba(args, 'unmod', model)
        tatoeba(args, 'whitened', model)
        tatoeba(args, 'cbie', model)

        # doing similar things for wiki dataset
        args = argparse.Namespace(model=model, layer=11, device='0', dataset='wiki', batch_size=64,
                                  save_whitened=True, save_cbie=True)

        extract_analyse(args)

        # doing similar things for STS
        args = argparse.Namespace(model=model, layer=11, device='0', dataset='sts', batch_size=64,
                                  save_whitened=True, save_cbie=True)
        extract_analyse(args)
        sts(args, 'unmod', model)
        sts(args, 'whitened', model)
        sts(args, 'cbie', model)


def extract_analyse(args):
    extract_sent_embeddings.main(args)
    # outlier/anisotropy analysis on unmodified embeds
    args.append_file_name = ""
    args.stdevs = 5
    args.type = 'all'  # should i be doing 'language' (as well)?
    print("ANALYZING UNMODIFIED EMBEDDINGS (finding outliers)")
    find_outliers.main(args)
    args.dim = -1
    print("ANALYZING UNMODIFIED EMBEDDINGS (anisotropy)")
    anisotropy.main(args)
    args.job = 'means'
    args.language = ''
    print("PLOTTING MEANS OF UNMODIFIED EMBEDDINGS")
    plots.main(args)
    # cbie version
    args.append_file_name = "_cbie"
    print("ANALYZING CBIE EMBEDDINGS (finding outliers)")
    find_outliers.main(args)
    print("ANALYZING CBIE EMBEDDINGS (anisotropy)")
    anisotropy.main(args)
    print("PLOTTING CBIE EMBEDDINGS")
    plots.main(args)
    # whitening version
    args.append_file_name = "_whitened"
    print("ANALYZING WHITENED EMBEDDINGS (finding outliers)")
    find_outliers.main(args)
    print("ANALYZING WHITENED EMBEDDINGS (anisotropy)")
    anisotropy.main(args)
    print("PLOTTING WHITENED EMBEDDINGS")
    plots.main(args)


def tatoeba(args, post_proc, model):
    args.predict_dir = f'../predictions/{model}/{post_proc}/tatoeba/'
    append = '' if post_proc == 'unmod' else f'_{post_proc}'
    for lang in langs_tatoeba_2:
        args.src_language = lang
        lang_3 = lang_dict[lang]
        src = torch.load(f'../embs/tatoeba/{model}/{args.layer}/{lang_3}/{lang_3}{append}.pt').numpy().astype(np.float32)
        tgt = torch.load(f'../embs/tatoeba/{model}/{args.layer}/{lang_3}/eng{append}.pt').numpy().astype(np.float32)
        predict_tatoeba(args, src, tgt)
    overall_scores, detailed_scores = evaluate(
        prediction_folder=f'../predictions/{model}/{post_proc}/', label_folder='../data/labels/'
    )
    overall_scores.update(detailed_scores)
    print(json.dumps(overall_scores))


STS_GOLD = {}

def _load_float_file(path):
    with open(path) as f_in:
        float_list = [float(line.strip()) for line in f_in]
    return np.array(float_list)


def _load_sts_gold():
    STS_GOLD.update({
        task: _load_float_file(path)
        for task, path in sts_gold_files.items()
    })


def sts(args, post_proc, model):
    if not STS_GOLD:
        _load_sts_gold()

    args.predict_dir = f'../predictions/{model}/{post_proc}/tatoeba/'
    append = '' if post_proc == 'unmod' else f'_{post_proc}'

    scores = {}
    for track, true_sim in STS_GOLD.items():
        lng1 = torch.load(f'../embs/sts/{model}/{args.layer}/{track}/lng1{append}.pt').numpy().astype(np.float32)
        lng2 = torch.load(f'../embs/sts/{model}/{args.layer}/{track}/lng2{append}.pt').numpy().astype(np.float32)
        model_sim = 1 - paired_cosine_distances(lng1, lng2)
        scores[track]= np.corrcoef(np.stack((model_sim, true_sim)))[0, 1]

    print(json.dumps(scores))


if __name__ == "__main__":
    main()
