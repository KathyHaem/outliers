# coding=utf-8
# Copyright The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Use pre-trained models for retrieval tasks and evaluate on them."""

import argparse

import collections
import logging
import os
import json
from typing import List, Tuple

import faiss
import numpy as np
import torch
from tqdm import tqdm

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    XLMConfig,
    XLMModel,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaModel,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoConfig, AutoTokenizer, AutoModel,
)

from scripts.third_party.utils_retrieve import mine_bitext, bucc_eval, similarity_search
from scripts.third_party.utils_lareqa import load_data as lareqa_load_data
from scripts.third_party.utils_lareqa import mean_avg_prec as lareqa_mean_avg_prec
from scripts.third_party.run_retrieval_qa import BertForSequenceRetrieval

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.keys()) for conf in (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                     XLM_PRETRAINED_CONFIG_ARCHIVE_MAP)), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlm": (XLMConfig, XLMModel, XLMTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "bert-retrieval": (BertConfig, BertForSequenceRetrieval, BertTokenizer)
}


# Dimensions to remove for BUCC task
# REMOVE_DIMS = [12, 63, 145, 151, 152, 266, 267, 459, 723, 728, 588, 306, 239, 184, 180]


def load_embeddings(embed_file, remove_dims=None):
    logger.info(' loading from {}'.format(embed_file))
    # embeds = np.load(embed_file, mmap_mode="r")
    embeds = np.load(embed_file)
    if remove_dims is not None:
        print(f"Removing dimensions {remove_dims}")
        for s in embeds:
            for dim in remove_dims:
                s[dim] = 0
    return embeds


def prepare_batch(sentences, tokenizer, model_type, device="cuda", max_length=512, langid=None,
                  use_local_max_length=True, pool_skip_special_token=False):
    pad_token = tokenizer.pad_token
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]

    batch_input_ids = []
    batch_attention_mask = []
    batch_pool_mask = []

    local_max_length = min(max([len(s) for s in sentences]) + 2, max_length)
    if use_local_max_length:
        max_length = local_max_length

    for sent in sentences:
        if len(sent) > max_length - 2:
            sent = sent[: (max_length - 2)]
        input_ids = tokenizer.convert_tokens_to_ids([cls_token] + sent + [sep_token])

        padding_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        pool_mask = [0] + [1] * (len(input_ids) - 2) + [0] * (padding_length + 1)
        input_ids = input_ids + ([pad_token_id] * padding_length)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_pool_mask.append(pool_mask)

    input_ids = torch.LongTensor(batch_input_ids).to(device)
    attention_mask = torch.LongTensor(batch_attention_mask).to(device)

    if pool_skip_special_token:
        pool_mask = torch.LongTensor(batch_pool_mask).to(device)
    else:
        pool_mask = attention_mask

    if model_type == "xlm":
        langs = torch.LongTensor([[langid] * max_length for _ in range(len(sentences))]).to(device)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "langs": langs}, pool_mask
    elif model_type == 'bert' or model_type == 'xlmr':
        token_type_ids = torch.LongTensor([[0] * max_length for _ in range(len(sentences))]).to(device)
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "return_dict": True}, pool_mask
    elif model_type == 'bert-retrieval':
        token_type_ids = torch.LongTensor([[0] * max_length for _ in range(len(sentences))]).to(device)
        return {"q_input_ids": input_ids, "q_attention_mask": attention_mask,
                "q_token_type_ids": token_type_ids}, pool_mask


def tokenize_text(text_file, tok_file, tokenizer, lang=None):
    if os.path.exists(tok_file):
        tok_sentences = [l.strip().split(' ') for l in open(tok_file, 'r', encoding='utf-8')]
        logger.info(' -- loading from existing tok_file {}'.format(tok_file))
        return tok_sentences

    tok_sentences = []
    sents = [l.strip() for l in open(text_file, 'r', encoding='utf-8')]
    with open(tok_file,  'w', encoding='utf-8') as writer:
        for sent in tqdm(sents, desc='tokenize'):
            if isinstance(tokenizer, XLMTokenizer):
                tok_sent = tokenizer.tokenize(sent, lang=lang)
            else:
                tok_sent = tokenizer.tokenize(sent)
            tok_sentences.append(tok_sent)
            writer.write(' '.join(tok_sent) + '\n')
    logger.info(' -- save tokenized sentences to {}'.format(tok_file))

    logger.info('============ First 5 tokenized sentences ===============')
    for i, tok_sentence in enumerate(tok_sentences[:5]):
        logger.info(f"S{i}: {''.join(tok_sentence)}")
    logger.info('==================================')
    return tok_sentences


def load_model(args, lang, output_hidden_states=None):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if output_hidden_states is not None:
        config.output_hidden_states = output_hidden_states
    langid = config.lang2id.get(lang, config.lang2id["en"]) if args.model_type == 'xlm' else 0
    logger.info("langid={}, lang={}".format(langid, lang))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)
    logger.info("tokenizer.pad_token={}, pad_token_id={}".format(tokenizer.pad_token, tokenizer.pad_token_id))
    if args.init_checkpoint:
        model = AutoModel.from_pretrained(args.init_checkpoint, config=config, cache_dir=args.init_checkpoint)
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    model.eval()
    return config, model, tokenizer, langid


def extract_embeddings_bucc_tatoeba(args, text_file, tok_file, embed_file, lang='en', pool_type='mean'):
    num_embeds = args.num_layers
    # if args.task_name == 'bucc2018': Warning: trying to save only layer 7 doesn't actually work right now;
    # it would actually end up saving layer 0 and pretending it's layer 7!!
    #    all_embed_files = ["{}_7.npy".format(embed_file)]  # only compute and save layer 7 embs for BUCC
    # else:
    #    all_embed_files = ["{}_{}.npy".format(embed_file, i) for i in range(num_embeds)]
    all_embed_files = [f"{embed_file}_{i}.npy" for i in range(num_embeds)]

    if all(os.path.exists(f) for f in all_embed_files):
        logger.info(f'loading files from {all_embed_files}')
        return [load_embeddings(f) for f in all_embed_files]

    config, model, tokenizer, langid = load_model(args, lang, output_hidden_states=True)

    sent_toks = tokenize_text(text_file, tok_file, tokenizer, lang)
    max_length = max([len(s) for s in sent_toks])
    logger.info(f'max length of tokenized text = {max_length}')

    batch_size = args.batch_size
    num_batch = int(np.ceil(len(sent_toks) * 1.0 / batch_size))
    num_sents = len(sent_toks)

    # if args.task_name == 'bucc2018':  # again, exception for BUCC
    #    all_embeds = [np.zeros(shape=(num_sents, args.embed_size), dtype=np.float32)]
    # else:
    #    all_embeds = [np.zeros(shape=(num_sents, args.embed_size), dtype=np.float32) for _ in range(num_embeds)]
    all_embeds = [np.zeros(shape=(num_sents, args.embed_size), dtype=np.float32) for _ in range(num_embeds)]
    print("allocated empty embedding matrix of shape {}, {}".format(len(all_embeds), all_embeds[0].shape))
    for i in tqdm(range(num_batch), desc='Batch'):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_sents)
        batch, pool_mask = prepare_batch(sent_toks[start_index: end_index],
                                         tokenizer,
                                         args.model_type,
                                         args.device,
                                         args.max_seq_length,
                                         langid=langid,
                                         pool_skip_special_token=args.pool_skip_special_token)

        with torch.no_grad():
            outputs = model(**batch)

            if args.model_type == 'bert' or args.model_type == 'xlmr':
                # batch_size, sequence_length, hidden_size
                all_layer_outputs: Tuple[torch.FloatTensor] = outputs["hidden_states"]
            elif args.model_type == 'xlm':
                _, all_layer_outputs = outputs

            # get the pool embedding
            if pool_type == 'cls':
                all_batch_embeds = cls_pool_embedding(all_layer_outputs[-args.num_layers:])
            else:
                all_batch_embeds = []
                all_layer_outputs = all_layer_outputs[-args.num_layers:]
                all_batch_embeds.extend(mean_pool_embedding(all_layer_outputs, pool_mask))

        for embeds, batch_embeds in zip(all_embeds, all_batch_embeds):
            embeds[start_index: end_index] = batch_embeds.cpu().numpy().astype(np.float32)
        del all_layer_outputs
        torch.cuda.empty_cache()

    if embed_file is not None:
        for file, embeds in zip(all_embed_files, all_embeds):
            logger.info('save embed {} to file {}'.format(embeds.shape, file))
            np.save(file, embeds)
        """if args.task_name == 'bucc2018':
            for file, embeds in zip([all_embed_files[7]], [all_embeds[7]]): ## modified to only consider and save layer 8 embs
                logger.info('save embed {} to file {}'.format(embeds.shape, file))
                np.save(file, embeds)
        else:
            for file, embeds in zip(all_embed_files, all_embeds): 
                logger.info('save embed {} to file {}'.format(embeds.shape, file))
                np.save(file, embeds)"""
    return all_embeds


def extract_encodings_wikinews_lareqa(args, text_file, tok_file, embed_file, lang='en', max_seq_length=None):
    """Get final encodings (not all layers, as extract_embeddings does)."""
    embed_file_path = f"{embed_file}.npy"
    if os.path.exists(embed_file_path):
        logger.info('loading file from {}'.format(embed_file_path))
        return load_embeddings(embed_file_path)

    config, model, tokenizer, langid = load_model(args, lang, output_hidden_states=False)

    sent_toks = tokenize_text(text_file, tok_file, tokenizer, lang)
    max_length = max([len(s) for s in sent_toks])
    logger.info('max length of tokenized text = {}'.format(max_length))

    batch_size = args.batch_size
    num_batch = int(np.ceil(len(sent_toks) * 1.0 / batch_size))
    num_sents = len(sent_toks)

    embeds = np.zeros(shape=(num_sents, args.embed_size), dtype=np.float32)
    for i in tqdm(range(num_batch), desc='Batch'):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_sents)

        # If given, custom sequence length overrides the args value.
        max_seq_length = max_seq_length or args.max_seq_length
        batch, pool_mask = prepare_batch(sent_toks[start_index: end_index],
                                         tokenizer,
                                         args.model_type,
                                         args.device,
                                         max_seq_length,
                                         langid=langid,
                                         pool_skip_special_token=args.pool_skip_special_token)
        with torch.no_grad():
            # TODO: add other retrieval baseline models
            if args.model_type == 'bert-retrieval':
                batch['inference'] = True
                outputs = model(**batch)
                batch_embeds = outputs
            else:
                logger.fatal("Unsupported model-type '%s' - "
                             "perhaps extract_embeddings() must be used?")

            embeds[start_index: end_index] = batch_embeds.cpu().numpy().astype(np.float32)
        torch.cuda.empty_cache()

    if embed_file is not None:
        logger.info('save embed {} to file {}'.format(embeds.shape, embed_file_path))
        np.save(embed_file_path, embeds)
    return embeds


def mean_pool_embedding(all_layer_outputs: Tuple[torch.FloatTensor], masks: torch.FloatTensor):
    """
      Args:
        all_layer_outputs: tuple of torch.FloatTensor, (B, L, D)
        masks: torch.FloatTensor, (B, L)
      Return:
        sent_emb: list of torch.FloatTensor, (B, D)
    """
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1)
        embeds = embeds / masks.sum(dim=1).view([-1, 1]).float()
        sent_embeds.append(embeds)
    return sent_embeds


def cls_pool_embedding(all_layer_outputs):
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = embeds[:, 0, :]
        sent_embeds.append(embeds)
    return sent_embeds


# TODO(jabot): Move this to a shared location for the task.
def el_eval(queries, predictions):
    """Evaluate entity linking.

    Args:
      queries: list of query dictionaries
      predictions: list of prediction lists
    Returns:
      mean-reciprocal rank
    """

    def _reciprocal_rank(labels):
        for rank, label in enumerate(labels, start=1):
            if label:
                return 1.0 / rank
        return 0.0

    assert len(queries) == len(predictions)

    reciprocal_ranks = []
    for query, prediction_list in zip(queries, predictions):
        ground_truth = query["entity_qid"]
        labels = [int(p == ground_truth) for p in prediction_list]
        reciprocal_ranks.append(_reciprocal_rank(labels))
    return np.mean(reciprocal_ranks)


def main(args):
    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logging.info("Input args: %r" % args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.task_name == 'bucc2018':
        eval_bucc(args)
    elif args.task_name == 'lareqa':
        eval_lareqa(args)
    elif args.task_name == 'wikinews_el':
        eval_wikinews(args)
    elif args.task_name == 'tatoeba':
        eval_tatoeba(args)


def eval_bucc(args):
    best_threshold = None
    SL, TL = args.src_language, args.tgt_language
    for split in ['dev', 'test']:
        prefix = os.path.join(args.output_dir, f'{SL}-{TL}.{split}')
        if args.extract_embeds:
            for lang in [SL, TL]:
                extract_embeddings_bucc_tatoeba(
                    args, text_file=f'{prefix}.{lang}.txt', tok_file=f'{prefix}.{lang}.tok',
                    embed_file=f'{prefix}.{lang}.emb', lang=lang)

        if args.mine_bitext:
            num_layers = args.num_layers
            if args.specific_layer != -1:
                indices = [args.specific_layer]
            else:
                indices = list(range(num_layers))

            for idx in indices:
                suf = str(idx)
                cand2score_file = os.path.join(
                    args.output_dir,
                    '{}_{}_{}.tsv'.format(args.candidate_prefix, suf, split))
                if os.path.exists(cand2score_file):
                    logger.info('cand2score_file {} exists'.format(cand2score_file))
                else:
                    x = load_embeddings(f'{prefix}.{SL}.emb_{idx}.npy', args.remove_dim)
                    y = load_embeddings(f'{prefix}.{TL}.emb_{idx}.npy', args.remove_dim)
                    mine_bitext(x, y, f'{prefix}.{SL}.txt', f'{prefix}.{TL}.txt', cand2score_file, dist=args.dist,
                                use_gpu=(torch.cuda.is_available() and not args.no_cuda),
                                use_shift_embeds=args.use_shift_embeds)
                # gold_file = f'{prefix}.gold'
                gold_file = f'../data/bucc2018/{SL}-en.{split}.gold'
                print("gold file name:", gold_file)

                """if os.path.exists(gold_file):
                    predict_file = os.path.join(args.predict_dir, f'test-{SL}.tsv')
                    print("predict file name:", predict_file)
                    results = bucc_eval(cand2score_file, gold_file, f'{prefix}.{SL}.txt', f'{prefix}.{TL}.txt',
                                        f'{prefix}.{SL}.id', f'{prefix}.id', predict_file, best_threshold)
                    best_threshold = results['best-threshold']
                    logger.info('--Candidates: {}'.format(cand2score_file))
                    logger.info(
                        'index={} '.format(suf) + ' '.join('{}={:.4f}'.format(k, v) for k, v in results.items()))"""

                predict_file = os.path.join(args.predict_dir, f'bucc2018/test-{SL}.tsv')
                # create_predict_file = open(predict_file, 'w+')
                print("predict file name:", predict_file)
                os.makedirs(os.path.dirname(predict_file), exist_ok=True)
                results = bucc_eval(cand2score_file, gold_file, f'{prefix}.{SL}.txt', f'{prefix}.{TL}.txt',
                                    f'{prefix}.{SL}.id', f'{prefix}.{TL}.id', predict_file, best_threshold)
                best_threshold = results['best-threshold']
                logger.info('--Candidates: {}'.format(cand2score_file))
                logger.info(
                    'index={} '.format(suf) + ' '.join('{}={:.4f}'.format(k, v) for k, v in results.items()))


def eval_lareqa(args):
    # The lareqa data is copied and stored here.
    base_path = os.path.join(args.data_dir, 'lareqa')
    dataset_to_dir = {
        "xquad": "xquad-r",
    }
    squad_dir = os.path.join(base_path, dataset_to_dir["xquad"])
    # Load the question set and candidate set.
    squad_per_lang = {}
    languages = set()
    # Load all files in the given directory, expecting names like 'en.json',
    # 'es.json', etc.
    for filename in os.listdir(squad_dir):
        language = os.path.splitext(filename)[0]
        languages.add(language)
        with open(os.path.join(squad_dir, filename), "r") as f:
            squad_per_lang[language] = json.load(f)
        print("Loaded %s" % filename)
    question_set, candidate_set = lareqa_load_data(squad_per_lang)
    # root directory where the outputs will be stored.
    root = args.output_dir
    question_ids_file = os.path.join(root, "question_uids.txt")
    question_embed_file = os.path.join(root, "question_encodings.npz")
    q_id2embed = {}
    for lang, questions_single_lang in question_set.by_lang.items():
        lang_file = os.path.join(root, lang + "-questions.txt")
        tok_file = os.path.join(root, lang + "-questions.tok")
        embed_file = os.path.join(root, lang + "-questions.emb")
        ids = []
        with open(lang_file, "w") as f:
            for question in questions_single_lang:
                f.write("%s\n" % question.question)
                ids.append(question.uid)

        # Array of [number of questions, embed_size]
        embeddings = extract_encodings_wikinews_lareqa(args, lang_file, tok_file, embed_file,
                                                       lang=lang,
                                                       max_seq_length=args.max_query_length)
        for i, embedding in enumerate(embeddings):
            q_id2embed[ids[i]] = embedding
            questions_single_lang[i].encoding = embedding
    with open(question_ids_file, "w") as f:
        f.writelines("%s\n" % id_ for id_ in list(q_id2embed.keys()))
    with open(question_embed_file, "wb") as f:
        np.save(f, np.array(list(q_id2embed.values())))
    # Do the same thing for the candidates.
    c_id2embed_sentences_and_contexts = {}
    for lang, candidates_single_lang in candidate_set.by_lang.items():
        lang_file_sentences_and_contexts = os.path.join(
            root, lang + "-candidates_sentences_and_contexts.txt")
        tok_file_sentences_and_contexts = os.path.join(
            root, lang + "-candidates_sentences_and_contexts.tok")
        embed_file_sentences_and_contexts = os.path.join(
            root, lang + "-candidates_sentences_and_contexts.emb")

        c_ids = []
        sentences_and_contexts = [
            (candidate.sentence + candidate.context).replace("\n", "")
            for candidate in candidates_single_lang]

        with open(lang_file_sentences_and_contexts, "w", encoding='utf-8') as f:
            f.write("\n".join(sentences_and_contexts))

        for candidate in candidates_single_lang:
            c_ids.append(candidate.uid)

        # Array of [number of candidates in single lang, embed_size]
        embeddings = extract_encodings_wikinews_lareqa(
            args,
            lang_file_sentences_and_contexts,
            tok_file_sentences_and_contexts,
            embed_file_sentences_and_contexts,
            lang=lang,
            max_seq_length=args.max_answer_length)

        for i, embedding in enumerate(embeddings):
            candidates_single_lang[i].encoding = {'sentences_and_contexts': embedding}
            c_id2embed_sentences_and_contexts[c_ids[i]] = embedding
    with open(os.path.join(root, "candidate_uids.txt"), "w") as f:
        f.writelines("%s\n" % id_ for id_ in list(c_id2embed_sentences_and_contexts.keys()))
    with open(os.path.join(root, "candidate_encodings_sentences_and_contexts.npz"), "wb") as f:
        np.save(f, np.array(list(c_id2embed_sentences_and_contexts.values())))
    print(question_set, candidate_set)
    print("*" * 10)
    lareqa_mean_avg_prec(question_set, candidate_set)
    print("*" * 10)


def eval_wikinews(args):
    base_output_path = os.path.join(args.output_dir, 'wikinews_el')

    def _read_jsonl(path):
        """Read jsonl record from path."""
        with open(path, "r") as fin:
            data = [json.loads(line) for line in fin]
            return data

    # Build a single index of all candidate vectors, populated on the-fly below.
    candidate_index = faiss.IndexFlatIP(args.embed_size)
    # Encode candidates.
    # Split by language for encoding phase only.
    candidates = _read_jsonl(os.path.join(args.data_dir, "candidates_1m.jsonl"))
    logger.info("Loaded {} candidates.".format(len(candidates)))
    qid2text = {c["qid"]: c["text"] for c in candidates}
    candidates_by_lang = collections.defaultdict(list)
    candidate_id2qid = []
    for c in candidates:
        candidates_by_lang[c["lang"]].append(c)
    for lang, candidates_slice in candidates_by_lang.items():
        candidate_text_file = os.path.join(args.output_dir,
                                           f"candidates.{lang}.txt")
        candidate_tok_file = os.path.join(args.output_dir,
                                          f"candidates.{lang}.tok")
        embed_file = os.path.join(args.output_dir, f"candidates.{lang}.emb")
        with open(candidate_text_file, "w") as f:
            for candidate in candidates_slice:
                f.write("%s\n" % candidate["text"])
                candidate_id2qid.append(candidate["qid"])

        candidate_encodings = extract_encodings_wikinews_lareqa(args, candidate_text_file,
                                                                candidate_tok_file,
                                                                embed_file, lang=lang,
                                                                max_seq_length=args.max_query_length)
        faiss.normalize_L2(candidate_encodings)
        candidate_index.add(candidate_encodings)
    logger.info("Candidate index size = %d" % candidate_index.ntotal)
    # Encode queries.
    # list of dictionaries.
    queries = _read_jsonl(os.path.join(args.data_dir,
                                       "wikinews15_v2_xtreme.jsonl"))
    logger.info("Loaded {} queries.".format(len(queries)))
    # Split input by languages in keeping with the other tasks.
    queries_by_lang = collections.defaultdict(list)
    for q in queries:
        queries_by_lang[q["lang"]].append(q)
    macro_mrr = []
    for lang, queries_slice in queries_by_lang.items():
        query_text_file = os.path.join(args.output_dir, f"queries.{lang}.txt")
        query_tok_file = os.path.join(args.output_dir, f"queries.{lang}.tok")
        embed_file = os.path.join(args.output_dir, f"queries.{lang}.emb")
        query_predictions_file = os.path.join(args.output_dir,
                                              f"queries.{lang}.predicted.txt")
        with open(query_text_file, "w") as f:
            for query in queries_slice:
                f.write("%s\n" % query["query"])

        query_encodings = extract_encodings_wikinews_lareqa(args, query_text_file, query_tok_file,
                                                            embed_file, lang=lang,
                                                            max_seq_length=args.max_query_length)

        # Retrieve K nearest-neighbors for each query in the current query slice,
        # using exact search.
        K = 100
        scores, predictions = candidate_index.search(query_encodings, K)
        predictions_qids = []
        with open(query_predictions_file, "w") as f:
            for i, neighbor_ids in enumerate(predictions):
                # Map predictions from internal sequential ids to entity QIDs.
                qids = [candidate_id2qid[neighbor] for neighbor in neighbor_ids]
                predictions_qids.append(qids)
                if i < 5:  # for debug inspection
                    logger.info("Retrieval output for '%s'" % queries_slice[i]["query"])
                    logger.info("gold=%s, predicted:" % queries_slice[i]["entity_qid"])
                    for qid in qids[:5]:
                        logger.info("\t%s: %s" % (qid, qid2text[qid]))
                f.write("%s\n" % "\t".join(qids))

        mrr_slice = el_eval(queries_slice, predictions_qids)
        macro_mrr.append(mrr_slice)
        logger.info("MRR for query_slice: %.4f" % mrr_slice)
    print("*" * 10)
    logger.info("Final MRR (macro-averaged over languages): %.4f" %
                np.mean(macro_mrr))
    print("*" * 10)


def eval_tatoeba(args):
    src_lang2 = args.src_language
    tgt_lang2 = args.tgt_language
    src_text_file = os.path.join(args.data_dir, f'{src_lang2}-en.{src_lang2}')
    tgt_text_file = os.path.join(args.data_dir, f'{src_lang2}-en.en')
    src_tok_file = os.path.join(args.output_dir, f'{src_lang2}-en.tok.{src_lang2}')
    tgt_tok_file = os.path.join(args.output_dir, f'{src_lang2}-en.tok.en')
    all_src_embeds = extract_embeddings_bucc_tatoeba(args, src_text_file, src_tok_file, None, lang=src_lang2)
    all_tgt_embeds = extract_embeddings_bucc_tatoeba(args, tgt_text_file, tgt_tok_file, None, lang=tgt_lang2)

    # ZERO OUT DIMENSIONS
    if args.remove_dim is not None:
        all_src_embeds, all_tgt_embeds = zero_dims(all_src_embeds, all_tgt_embeds, args.remove_dim)
    if args.extract_rankings:
        extract_rankings(all_src_embeds, all_tgt_embeds, args, src_lang2, tgt_lang2)
    for i in [args.specific_layer]:
        x, y = all_src_embeds[i], all_tgt_embeds[i]
        predict_tatoeba(args, x, y)


def predict_tatoeba(args, src_embs, tgt_embs):
    predictions = similarity_search(src_embs, tgt_embs, args.embed_size, normalize=(args.dist == 'cosine'))
    os.makedirs(os.path.dirname(args.predict_dir), exist_ok=True)
    with open(os.path.join(args.predict_dir, f'test-{args.src_language}.tsv'), 'w', encoding='utf-8') as fout:
        for p in predictions:
            fout.write(str(p) + '\n')
    # print(f"finished writing predictions for {args.src_language}-{args.tgt_language}")


def extract_rankings(all_src_embeds, all_tgt_embeds, args, src_lang2, tgt_lang2):
    for i in [args.specific_layer]:
        x, y = all_src_embeds[i], all_tgt_embeds[i]
        predictions, scores = similarity_search(x, y, args.embed_size, normalize=(args.dist == 'cosine'),
                                                extract_rankings=True)
        os.makedirs(os.path.join(args.predict_dir, "ids"), exist_ok=True)
        with open(os.path.join(args.predict_dir + "ids", f'test-{src_lang2}.json'), 'w', encoding='utf-8') as fout:
            preds = json.dumps(predictions.tolist())
            fout.write(preds)
        os.makedirs(os.path.join(args.predict_dir, "cosines"), exist_ok=True)
        json_filename = os.path.join(args.predict_dir + "cosines", f'test-cosines-{src_lang2}.json')
        with open(json_filename, 'w', encoding='utf-8') as wout:
            scrs = json.dumps(scores.tolist())
            wout.write(scrs)
        print(f"finished writing predictions for {src_lang2}-{tgt_lang2}")


def zero_dims(all_src_embeds, all_tgt_embeds, remove_dim: List[int]):
    dimensions = [int(x) for x in remove_dim]
    print(f"MODIFYING DIMENSION(S) {dimensions}.")
    # take layer 8 for xlm-r (base)
    for sent in range(len(all_src_embeds[7])):
        for d in dimensions:
            all_src_embeds[7][sent][d] = 0
            # all_src_embeds[7][sent][d] = all_src_embeds[7][sent][d] / 15 # this was done for experiments with re-scaling
    for sent in range(len(all_tgt_embeds[7])):
        for d in dimensions:
            all_tgt_embeds[7][sent][d] = 0
            # all_tgt_embeds[7][sent][d] = all_tgt_embeds[7][sent][d] / 15
    return all_src_embeds, all_tgt_embeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BUCC bitext mining')
    parser.add_argument('--encoding', default='utf-8',
                        help='character encoding for input/output')
    parser.add_argument('--src_file', default=None, help='src file')
    parser.add_argument('--tgt_file', default=None, help='tgt file')
    parser.add_argument('--gold', default=None,
                        help='File name of gold alignments')
    parser.add_argument('--threshold', type=float, default=-1,
                        help='Threshold (used with --output)')
    parser.add_argument('--embed_size', type=int, default=768,
                        help='Dimensions of output embeddings')
    parser.add_argument('--pool_type', type=str, default='mean',
                        help='pooling over work embeddings')

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the input files for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list"
    )
    parser.add_argument("--src_language", type=str, default="en", help="source language.")
    parser.add_argument("--tgt_language", type=str, default="de", help="target language.")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
    parser.add_argument("--tgt_text_file", type=str, default=None, help="tgt_text_file.")
    parser.add_argument("--src_text_file", type=str, default=None, help="src_text_file.")
    parser.add_argument("--tgt_embed_file", type=str, default=None, help="tgt_embed_file")
    parser.add_argument("--src_embed_file", type=str, default=None, help="src_embed_file")
    parser.add_argument("--tgt_tok_file", type=str, default=None, help="tgt_tok_file")
    parser.add_argument("--src_tok_file", type=str, default=None, help="src_tok_file")
    parser.add_argument("--tgt_id_file", type=str, default=None, help="tgt_id_file")
    parser.add_argument("--src_id_file", type=str, default=None, help="src_id_file")
    parser.add_argument("--num_layers", type=int, default=12, help="num layers")
    parser.add_argument("--candidate_prefix", type=str, default="candidates")
    parser.add_argument("--pool_skip_special_token", action="store_true")
    parser.add_argument("--dist", type=str, default='cosine')
    parser.add_argument("--use_shift_embeds", action="store_true")
    parser.add_argument("--extract_embeds", action="store_true")
    parser.add_argument("--mine_bitext", action="store_true")
    parser.add_argument("--predict_dir", type=str, default=None, help="prediction folder")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory",
    )
    parser.add_argument("--log_file", default="train", type=str, help="log file")

    parser.add_argument(
        "--task_name",
        default="bucc2018",
        type=str,
        required=True,
        help="The task name",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=92,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--unify", action="store_true", help="unify sentences"
    )
    parser.add_argument("--split", type=str, default='training', help='split of the bucc dataset')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--specific_layer", type=int, default=7, help="use specific layer")
    parser.add_argument("--remove_dim", type=int, nargs='*', default=None,
                        help="dimensions to zero out, i.e. outlier dimension")
    parser.add_argument('--extract_rankings', action='store_true', help='different mode for tatoeba eval')
    args = parser.parse_args()

    main(args)
