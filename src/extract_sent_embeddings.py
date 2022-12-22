""" turning the colab file into a script """
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from constants import langs_tatoeba, langs_wiki
from post_processing import whitening, cluster_based


def read_tatoeba_data(target):
    # read target lang data
    with open(f'../data/tatoeba-parallel/tatoeba.{target}-eng.{target}', 'r', encoding='utf-8') as tgt:
        tgt_sents = [line for line in tgt.read().split("\n") if line]
    # read eng data
    with open(f'../data/tatoeba-parallel/tatoeba.{target}-eng.eng', 'r', encoding='utf-8') as eng:
        eng_sents = [line for line in eng.read().split("\n") if line]
    return tgt_sents, eng_sents


def mean_pooling(token_embeddings, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeds(data, model, tokenizer, args, device):
    tgt_embeddings_layer = None
    attention_masks = None

    dataloader = DataLoader(data, batch_size=args.batch_size, drop_last=False)
    max_len = max([len(x) for x in tokenizer(data, padding=True, truncation=True)['input_ids']])

    for batch in dataloader:
        encoded = tokenizer(batch, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        encoded.to(device)
        if attention_masks is None:
            attention_masks = encoded['attention_mask'].cpu()
        else:
            attention_masks = torch.cat((attention_masks, encoded['attention_mask'].cpu()))
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
            if tgt_embeddings_layer is None:
                tgt_embeddings_layer = output[2][1:][args.layer].cpu()
            else:
                tgt_embeddings_layer = torch.cat((tgt_embeddings_layer, output[2][1:][args.layer].cpu()))
    return mean_pooling(tgt_embeddings_layer, attention_masks)


def load_model(args, device):
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(device)
    model.eval()
    return model, tokenizer


def main(args):
    try:
        int(args.device)
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    except ValueError:
        device = args.device

    if args.dataset == "tatoeba":
        langs = langs_tatoeba
    elif args.dataset == "wiki":
        langs = langs_wiki

    else:
        raise ValueError("unknown dataset argument")

    model, tokenizer = load_model(args, device)

    if args.dataset == 'tatoeba':
        for lang in langs:
            print(f"Currently processing {lang}...")
            os.makedirs(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}', exist_ok=True)

            tgt, eng = read_tatoeba_data(lang)
            tgt_embeddings = get_embeds(tgt, model, tokenizer, args, device)
            eng_embeddings = get_embeds(eng, model, tokenizer, args, device)
            torch.save(tgt_embeddings, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')
            torch.save(eng_embeddings, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng.pt')
            if args.save_whitened:
                tgt_whitened = torch.Tensor(whitening(tgt_embeddings.numpy()))
                eng_whitened = torch.Tensor(whitening(eng_embeddings.numpy()))
                torch.save(tgt_whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_whitened.pt')
                torch.save(eng_whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng_whitened.pt')
            if args.save_cbie:
                n_cluster = max(len(eng) // 300, 1) 
                tgt_cbie = torch.Tensor(cluster_based(
                    tgt_embeddings.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=tgt_embeddings.shape[1]))
                eng_cbie = torch.Tensor(cluster_based(
                    eng_embeddings.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=eng_embeddings.shape[1]))
                torch.save(tgt_cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_cbie.pt')
                torch.save(eng_cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng_cbie.pt')
            print(f"Finished saving embeddings for {lang} in model {args.model}.")

    elif args.dataset == 'wiki':
        # Loading Wikipedia datasets
        df_su = pd.read_csv('../data/Wikipedia/Sundanese.csv', sep=',')
        df_sw = pd.read_csv('../data/Wikipedia/Swahili.csv', sep=',')
        df_en = pd.read_csv('../data/Wikipedia/English.csv', sep=',')
        df_es = pd.read_csv('../data/Wikipedia/Spanish.csv', sep=',')
        df_ar = pd.read_csv('../data/Wikipedia/Arabic.csv', sep=',')
        df_tr = pd.read_csv('../data/Wikipedia/Turkish.csv', sep=',')
        dfs = [df_ar, df_en, df_es, df_su, df_sw, df_tr]
        for lang, df in zip(langs, dfs):
            print(f"processing {lang}...")
            rep = get_embeds(df['Sentence'].tolist(), model, tokenizer, args, device)
            os.makedirs(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/', exist_ok=True)
            torch.save(rep, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')
            if args.save_whitened:
                whitened = torch.Tensor(whitening(rep.numpy()))
                torch.save(whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_whitened.pt')
            if args.save_cbie:
                n_cluster = max(rep.shape[0] // 300, 1)
                cbie = torch.Tensor(
                    cluster_based(rep.numpy, n_cluster=n_cluster, n_pc=12, hidden_size=rep.shape[1]))
                torch.save(cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_cbie.pt')
            print(f"Finished saving embeddings for {lang} in model {args.model}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Establish outlier dimensions.')
    parser.add_argument('--model', type=str, help="name of the model")
    parser.add_argument('--layer', type=int, help="which model layer to save embeddings are from")
    parser.add_argument('--device', type=str, default="0", help="which GPU/device to use")
    parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki"],
                        help="which dataset to encode (tatoeba, wiki)")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for encoding')
    parser.add_argument('--save_whitened', action='store_true', help='save embeddings processed with whitening as well')
    parser.add_argument('--save_cbie', action='store_true', help='save embeddings processed with cbie as well')
    args = parser.parse_args()
    main(args)
