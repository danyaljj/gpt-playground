import random

import torch
from tqdm import tqdm
from datasets import load_dataset
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder

# wandb.init(project='discrete prompt from continuous')

device = 'cuda'
model_size = "danyaljj/opengpt2_pytorch_backward"

model_backward = OpenGPT2LMHeadModel.from_pretrained(model_size)
model_backward.to(device)
model_backward.eval()

# tokenizer
encoder = Encoder()

if False:
    right_items = [
        "Who is the US president?",
        "How is the weather tomorrow?",
        "When will you go to the school?",
        "What time will the class begin?",
        "Where is the lecture?",
        "Why do I have a headache this week?"
    ]
else:
    dataset = load_dataset("squad")
    # for x in dataset['train']:
    #     print(x)
    right_items = [f" Question: {x['question']} - Answer: {x['answers']['text'][0]}"  for x in dataset['train'] if len(x['question']) < 1024 * 4]
    random.shuffle(right_items)
    # right_items = right_items[:10]
    # right_items = right_items[:100]
    right_items = right_items


def discrete_prompt_from_endings(right_items_ids, verbose = True):
    max_len = 10
    if False:
        # backward generation using one right response
        for right_item_ids in right_items_ids:
            print("  >>>>>>>>>>>>>>>>>>>>>>>>  ")
            output = model_backward.generate(right_item_ids)
            output_text = encoder.decode(output.tolist()[0][::-1])
            print(f"instance prompt: {output_text}")

    # backward generation using my for loop
    if False:
        all_logit_dists = None
        for right_item_ids in right_items_ids:
            logits_so_far = None
            for i in range(0, max_len):
                logits = model_backward(input_ids=right_item_ids).logits
                logits = logits[:, -1, :]
                logits = logits.unsqueeze(1)
                logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
                next_token = torch.argmax(logits)
                right_item_ids = torch.cat([right_item_ids, next_token.unsqueeze(0).unsqueeze(0)], 1)
            output_text = encoder.decode(right_item_ids.tolist()[0][::-1])
            print(f"instance prompt #2: {output_text}")
            if all_logit_dists == None:
                all_logit_dists = logits_so_far.cpu()
            else:
                all_logit_dists += logits_so_far.cpu()

        print(" ><><><><><>< Aggregated logits: ")
        right_item_ids = torch.LongTensor([[]])
        for i in range(0, max_len):
            logits = all_logit_dists[0, i, :]
            next_token = torch.argmax(logits)
            right_item_ids = torch.cat([right_item_ids, next_token.unsqueeze(0).unsqueeze(0)], 1)
        output_text = encoder.decode(right_item_ids.tolist()[0][::-1])
        print(f"Aggregated prompt: {output_text}")

    # backward generation using my for loop
    if True:
        generation_thus_far = torch.LongTensor([[]])
        for i in tqdm(range(0, max_len)):
            all_logit_dists = None
            for right_item_ids in right_items_ids:
                torch.cuda.empty_cache()
                new_right_ids = torch.cat([right_item_ids, generation_thus_far.to(device)], 1)
                logits = model_backward(input_ids=new_right_ids).logits.clone().detach()
                logits = logits[:, -1, :]
                logits = logits.unsqueeze(1)
                if all_logit_dists == None:
                    all_logit_dists = logits.cpu()
                else:
                    all_logit_dists += logits.cpu()
            next_token = torch.argmax(all_logit_dists)
            generation_thus_far = torch.cat(
                [
                    generation_thus_far,
                    next_token.unsqueeze(0).unsqueeze(0).clone().detach()
                ], 1
            )
        output_text = encoder.decode(right_item_ids.tolist()[0][::-1])
        print(f"Generated prompt: {output_text}")

def experiment0():
    '''
    test out the right to left moodel
    '''
    input = right_items[0]
    input_ids = encoder.encode(input)
    input_ids = torch.tensor([input_ids[::-1] ], dtype=torch.int).to(device)
    print(input_ids)

    output = model_backward.generate(input_ids)
    output_text = encoder.decode(output.tolist()[0][::-1])

    print(output_text)

def experiment1():
    right_item_ids = []
    for i, right_item in enumerate(right_items):
        if i < 20:
            print(right_item)
        input_ids = encoder.encode(right_item)
        input_ids = torch.tensor([input_ids[::-1] ], dtype=torch.int).to(device)
        right_item_ids.append(input_ids)
    for count in [10, 100, 1000, 10000, 100000]:
        print(" - - - - - - - ")
        print(f" => count: {count}")
        discrete_prompt_from_endings(right_item_ids[:count])



from google_ngram_downloader import readline_google_store
import json

def experiment_unigrams():
    # extract counts for GPT vocabulary
    # count_map = {}
    fout = open("unigram_file.jsonl", "w")
    counter = 0
    for fname, url, records in readline_google_store(ngram_len=1):
        print(fname)
        print(url)
        for r in records:
            text = r.ngram
            year = r.year
            if year < 2005 or year > 2008:
                continue
            match_count = r.match_count
            volume_count = r.volume_count
            indices = encoder.encoder.encode(text)
            if len(indices) == 1:
                counter += 1
                line= {
                    'idx': indices[0],
                    'text': text,
                    'year': year,
                    'match_count': match_count,
                    'volume_count': volume_count,
                }
                fout.write(json.dumps(line) + "\n")
                if counter % 1000 == 0:
                    print(f" * map size: {counter} -> {line}")




experiment0()
# experiment1()
# experiment_unigrams()


