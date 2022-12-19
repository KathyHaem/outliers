from collections import defaultdict
import torch
import argparse

# Code used for establishing expanded set of outliers.

parser = argparse.ArgumentParser(description='Find dimensions for additional tatoeba runs.')
parser.add_argument('model', type=str, help="name of the model (xlm-r, x2s-cca or x2s-mse)")
parser.add_argument('layer', type=int, help="which model layer the embeddings are from")
args = parser.parse_args()

langs = ['ara', 'heb', 'vie', 'ind', 'jav', 'tgl', 'eus', 'mal',
         'tel', 'afr', 'nld', 'deu', 'ell', 'ben', 'hin', 'mar',
         'urd', 'tam', 'fra', 'ita', 'por', 'spa', 'bul', 'rus',
         'jpn', 'kat', 'kor', 'tha', 'swh', 'cmn', 'kaz', 'tur',
         'est', 'fin', 'hun', 'pes']

farthest = defaultdict(float)

for lang in langs:
    print(f"Considering language {lang}.")
    TGT = lang
    target_emb = torch.load(f'embs_layer_{args.layer}/{args.model}/{TGT}/{TGT}.pt')
    tgt_mean = torch.abs(torch.mean(target_emb, axis=0))

    for i in torch.topk(tgt_mean, 10).indices:
        print(i)
        farthest[i.item()] += 1
    print("#######################")

print()
print("######## Dimensions farthest from zero: ########")
print(sorted(farthest.items(), key=lambda item: item[1], reverse=True))
