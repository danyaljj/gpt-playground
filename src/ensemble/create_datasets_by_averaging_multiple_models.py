# MSCLAR note: code extracted from https://huggingface.co/blog/how-to-generate

import argparse
import json
import os
import random
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset
from transformers import EvalPrediction, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, TopPLogitsWarper

discrimination_data_path = 'grover-models/discrimination-data'
output_dir = 'generated-datasets'
data_dir = '/gscratch/xlab/msclar/'

if not torch.cuda.is_available():
    data_dir = './'
    discrimination_data_path = './'

cache_dir = os.path.join(data_dir, '.cache')

title_max_size = 100
body_max_size = 200

MODELS_TO_EXECUTE = [
    "arwen-gpt2-medium-x21",
    "beren-gpt2-medium-x49",
    # "celebrimbor-gpt2-medium-x81",
    "durin-gpt2-medium-x343"
]

def combining_models(models, input_ids, target_ids):
    logits = []
    for model_name in models:
        output = models[model_name](input_ids=input_ids, labels=target_ids)
        logits.append(output.logits)
    return torch.stack(logits, dim=-1).sum(dim=-1) * (1 / len(models))


# https://github.com/huggingface/transformers/blob/56f50590d5a9ac881db9ee1753f4642cf3d33d28/src/transformers/models/gpt2/modeling_gpt2.py
loss_fct = torch.nn.CrossEntropyLoss()
def compute_cross_entropy_loss(lm_logits, labels):
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return loss


def huggingface_perplexity_fn(models_dict, model, encodings, device, combine_several_models=False):
    max_length = model.config.n_positions if model else list(models_dict.values())[0].config.n_positions
    stride = 64

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            if combine_several_models:
                outputs = combining_models(models_dict, input_ids, target_ids)
                loss = compute_cross_entropy_loss(outputs, target_ids)
            else:
                outputs = model(input_ids, labels=target_ids)
                loss = compute_cross_entropy_loss(outputs.logits, target_ids)

            neg_log_likelihood = loss * trg_len  # outputs[0] = loss

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl


def perplexity_experiment(args):
    args.average_logits = True  # the version that generated more reasonable text
    args.average_probs = False

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    for model_name in MODELS_TO_EXECUTE:
        models[model_name] = AutoModelForCausalLM.from_pretrained(
            "stanford-crfm/" + model_name, return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id, cache_dir=cache_dir)
        models[model_name].to(device)

    dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, args.input_document),
        field="data",
        cache_dir=cache_dir
    )['train']

    # MSCLAR: edits to be FAST
    dataset = dataset.filter(lambda x: x['split'] != 'test')
    dataset = dataset.filter(lambda e, i: i < len(dataset) // 2, with_indices=True)
    dataset = dataset.filter(lambda x: x['label'] == 'human')  # to have source articles to measure :)
    print(dataset)

    encodings = tokenizer("\n\n".join(dataset["article"]), return_tensors="pt")

    metrics = {}
    for model_name in MODELS_TO_EXECUTE:
        ppl = huggingface_perplexity_fn(None, models[model_name], encodings, device, combine_several_models=False)
        print('IndivPerplexity', model_name, ':', ppl.item())
        metrics[f'{model_name}_ppl'] = ppl.item()

    ppl = huggingface_perplexity_fn(models, None, encodings, device, combine_several_models=True)
    print('CombinedPerplexity:', ppl.item())
    metrics[f'combined_ppl'] = ppl.item()

    """
    # FIXME average perplexity per article to not OOM
    encodings = tokenizer("\n\n".join(dataset["article"]), return_tensors="pt")
    for model_name in MODELS_TO_EXECUTE:
        ppl = 0
        total = 0
        for row in dataset["article"]:
            encodings = tokenizer(row, return_tensors="pt")
            ppl += huggingface_perplexity_fn(None, models[model_name], encodings, device, combine_several_models=False).item()
            total += 1
        print('IndivPerplexity', model_name, ':', ppl / total)
    ppl = huggingface_perplexity_fn(models, None, encodings, device, combine_several_models=True)
    print('CombinedPerplexity:', ppl.item())
    """

    out = open('metrics.json', 'w')
    print(json.dumps(metrics, indent=4, sort_keys=True))
    json.dump(metrics, out)
    out.close()


def create_dataset_averaging_logits(args):
    assert args.top_p

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    if args.model_type.startswith('gpt2') or args.model_type.startswith('EleutherAI') or args.model_type.startswith('stanford-crfm'):
        tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    for model_name in MODELS_TO_EXECUTE:
        models[model_name] = AutoModelForCausalLM.from_pretrained(
            "stanford-crfm/" + model_name, return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id, cache_dir=cache_dir)
        models[model_name].to(device)

    dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, discrimination_data_path, 'base', f'p=0.96.jsonl'),
        cache_dir=cache_dir
    )['train']
    # MSCLAR: edits to be FAST
    dataset = dataset.filter(lambda x: x['split'] != 'test')
    dataset = dataset.filter(lambda e, i: i < len(dataset) // 2, with_indices=True)

    dataset = dataset.filter(lambda x: x['label'] == 'machine')
    # dataset = dataset.map(lambda example: {"text": example['title'] + ". "})

    generated_texts = []
    logits_warper = TopPLogitsWarper(top_p=args.top_p)
    for idx, row in enumerate(dataset):
        print(idx)
        if idx > args.sample_size > 0:
            break
        tokenized = tokenizer(row['title'] + ". ", max_length=title_max_size, truncation=True, return_tensors="pt")
        title_length = tokenized['input_ids'].shape[1]
        print('title_length', title_length)
        input_ids = tokenized['input_ids'].to(device)

        for _ in range(body_max_size):
            output = {}
            next_token_logits = []
            for model_name in models:
                output[model_name] = models[model_name](input_ids=input_ids) #, attention_mask=attention_mask)
                next_token_logits.append(output[model_name].logits[:, -1, :])

            assert args.average_logits and args.average_probs == False, f"These two options average_logits={args.average_logits} and average_probs={args.average_probs} can't be true at the same time! "

            if args.average_logits:
                next_token_scores = sum(next_token_logits)
            elif args.average_probs:
                # compute proba distribution for all, then sum
                next_token_scores = sum(logits.softmax(dim=-1) for logits in next_token_logits)
            else:
                raise Exception("unknown setting . . . ")

            # pre-process distribution
            # next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if next_tokens.item() == 50256:
                break

        row['article'] = tokenizer.decode(input_ids[0, title_length:], skip_special_tokens=True)
        generated_texts.append(row)

    filename = 'averaged_logits' if args.average_logits else 'average_probs'
    with open(os.path.join(data_dir, output_dir, f'{filename}_gpt2_medium.jsonl'), 'w') as outfile:
        outfile.write('\n')
        for entry in generated_texts:
            json.dump(entry, outfile)
            outfile.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--model_type', type=str, default='gpt2')
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--sample_size', type=int, default=-1)
    parser.add_argument('--average_logits', action='store_true')
    parser.add_argument('--average_probs', action='store_true')
    # parser.add_argument('--combine_several_models', action='store_true')
    parser.add_argument('--input_document', type=str)

    args = parser.parse_args()
    # create_dataset_averaging_logits(args)
    perplexity_experiment(args)