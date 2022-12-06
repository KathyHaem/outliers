""" turning the colab file into a script """
import argparse
import os

from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Establish outlier dimensions.')
parser.add_argument('--model', type=str, help="name of the model")
parser.add_argument('--layer', type=int, help="which model layer to save embeddings are from")
parser.add_argument('--device', type=str, default="0" help="which GPU/device to use")
parser.add_argument('--batch_size', type=int, default=128, help='batch size for encoding')
args = parser.parse_args()

try:
    int(args.device)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
except ValueError:
    device = args.device


def read_data(target):
    # read target lang data
    with open(f'../../data/tatoeba-parallel/tatoeba.{target}-eng.{target}', 'r') as tgt:
        tgt_sents = tgt.read().split("\n")
    # read eng data
    with open(f'../../data/tatoeba-parallel/tatoeba.{target}-eng.eng', 'r') as eng:
        eng_sents = eng.read().split("\n")
    return tgt_sents, eng_sents


def mean_pooling(token_embeddings, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


langs = ['ara', 'heb', 'vie', 'ind', 'jav', 'tgl', 'eus', 'mal', 'tel',
         'afr', 'nld', 'deu', 'ell', 'ben', 'hin', 'mar', 'urd', 'tam',
         'fra', 'ita', 'por', 'spa', 'bul', 'rus', 'jpn', 'kat', 'kor',
         'tha', 'swh', 'cmn', 'kaz', 'tur', 'est', 'fin', 'hun', 'pes']

model = AutoModel.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model.to(device)
model.eval()


def get_embeds(data):
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


for lang in langs:
    TGT = lang
    print(f"Currently processing {TGT}...")
    tgt, eng = read_data(TGT)

    tgt_embeddings = get_embeds(tgt)
    eng_embeddings = get_embeds(eng)

    os.makedirs(f'../embs/{args.model}/{args.layer}/{TGT}', exist_ok=True)
    torch.save(tgt_embeddings, f'../embs/{args.model}/{args.layer}/{TGT}/{TGT}.pt')
    torch.save(eng_embeddings, f'../embs/{args.model}/{args.layer}/{TGT}/eng.pt')
    print(f"Finished saving embeddings for {TGT} in model {args.model}.")
