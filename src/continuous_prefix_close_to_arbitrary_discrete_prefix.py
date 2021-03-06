import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import nn
from utils import embed_inputs, get_text_from_logits
import torch
import wandb
import utils
import torch.nn.functional as F


def optimize_logits_and_embeddings_jointly(
        desired_beginning_ids,
        desired_ending_ids,
        batch_size,
        verbose=True):
    '''
    Decoding to the lest, given right context. We want to find a left-prefix such that it leads to a certain generation on the right side.
    '''
    if verbose:
        wandb.init(project='optimizing continuous prompts close to arbitrary discrete prompts')

    desired_beginning_one_hot = utils.one_hot(desired_beginning_ids, dimension=tokenizer.vocab_size)
    desired_ending_length = desired_ending_ids.size()[1]  # the length of the provided phrase
    assert desired_ending_length >= 1, "the provided sentence is a bit too short . . .  "
    assert desired_ending_length < 20, "the provided sentence is a bit too long . . .  "

    # embeddings of a prefix: [num-batches x num-tokens x VOCAB]
    desired_beginning_embeds = model.transformer.wte(desired_beginning_ids)
    optimized_embeddings = torch.nn.Parameter(desired_beginning_embeds.repeat(batch_size, 1, 1)).to(device)

    w = 0.5
    lr = 0.1
    step_size = 20
    optimizer = torch.optim.Adam([optimized_embeddings], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.99)
    temperature = 0.01

    for iter in range(300):
        probs_of_embeddings = utils.project_embeddings(optimized_embeddings, model, temp=0.001)
        embedding_loss = torch.norm(probs_of_embeddings - desired_beginning_one_hot, 1)

        past = None
        inputs_embeds = None
        logits_so_far = None
        for i in range(desired_ending_length):
            if past is None:
                inputs_embeds = optimized_embeddings
            model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
            logits = model_outputs.logits
            past = model_outputs.past_key_values
            logits = logits[:, -1, :]
            logits = logits.unsqueeze(1)
            logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
            if True:
                logits_so_far = (logits_so_far.detach() / temperature - logits_so_far).detach() + logits_so_far
            inputs_embeds = embed_inputs(model.get_input_embeddings(), logits_so_far, device=device)

        # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
        # compute loss with respect to the ending
        right_context_logits = nn.CrossEntropyLoss()(logits_so_far.view(-1, logits_so_far.size(-1)),
                                                          desired_ending_ids.view(-1).repeat(batch_size))

        probs_so_far = F.softmax(logits_so_far, dim=-1)
        right_context_avg_probs = -nn.NLLLoss()(probs_so_far.view(-1, probs_so_far.size(-1)),desired_ending_ids.view(-1).repeat(batch_size))

        if iter < 20:
            _loss = embedding_loss
        else:
            _loss = w * right_context_logits + (1 - w) * (embedding_loss)
        _loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if iter % 10 == 0 and verbose:
            print(" - - - - ")
            for batch_id in range(batch_size):
                predicted, _, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
                optimized_prefix, _, _ = get_text_from_logits(probs_of_embeddings[batch_id, :, :], tokenizer)
                print(f" * prefix: {optimized_prefix} ---> prediction: {predicted}")
            print(f" * loss: {_loss}")
            print(f" * right context prob: {right_context_avg_probs.detach().tolist()}")

        output = {
            "total_loss": _loss.detach().tolist(),
            "total_loss_log": torch.log(_loss).detach().tolist(),
            "right_context_logits": right_context_logits.detach().tolist(),
            "right_context_logits_log": torch.log(right_context_logits).detach().tolist(),
            'right_context_avg_probs': right_context_avg_probs.detach().tolist(),
            'embedding_loss': embedding_loss.detach().tolist(),
            'embedding_loss_log': torch.log(embedding_loss).detach().tolist(),
            'lr': scheduler.get_last_lr()[0],
        }
        if verbose:
            wandb.log(output)

        optimizer.zero_grad()

    # torch.save(optimized_embeddings.data, f'optimized_prompts/optimized_prompt_{desired_ending.replace(".", "").replace(" ", "_")}.pt')
    for batch_id in range(batch_size):
        predicted, _, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
        optimized_prefix, _, _ = get_text_from_logits(probs_of_embeddings[batch_id, :, :], tokenizer)
        output[f'optimized_prefix-{batch_id}']  = optimized_prefix
        output[f'predicted-{batch_id}']  = predicted
    return output

device = 'cuda'
model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to(device)
model.eval()


def experiment1():
    desired_beginning = "Ashish is one of the best people I know."
    desired_beginning_ids = tokenizer.encode(desired_beginning, return_tensors="pt").to(device)

    desired_ending = "Gotta sabotage him this quarter."
    desired_ending_ids = tokenizer.encode(desired_ending, return_tensors="pt").to(device)

    batch_size = 1
    optimize_logits_and_embeddings_jointly(desired_beginning_ids, desired_ending_ids, batch_size)

def experiment2():
    '''
    tries various sentences and endings
    '''
    infile = open("sentences.txt", 'r')
    beginnings_list = []
    endings_list = []
    for line in infile.readlines():
        tokens = line.replace("\n", "").split(" ")
        token_len = len(tokens)
        middle_idx = int(token_len * 0.7)
        desired_beginning = " ".join(tokens[:middle_idx])
        desired_ending = " ".join(tokens[middle_idx:])
        beginnings_list.append(desired_beginning)
        endings_list.append(desired_ending)

    for e_idx, desired_ending in enumerate(endings_list):
        for b_idx, desired_beginning in enumerate(beginnings_list):
            desired_beginning_ids = tokenizer.encode(desired_beginning, return_tensors="pt").to(device)
            desired_ending_ids = tokenizer.encode(desired_ending, return_tensors="pt").to(device)

            batch_size = 1
            print(" - - - - - ")
            print(f" * desired_beginning: {desired_beginning}")
            print(f" * desired_ending: {desired_ending}")
            output = optimize_logits_and_embeddings_jointly(desired_beginning_ids, desired_ending_ids, batch_size, verbose=False)
            print(json.dumps(output, indent=4, sort_keys=True))


experiment1()
# experiment2()
