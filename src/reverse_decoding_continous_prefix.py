from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from torch import nn
import wandb
import math
from utils import embed_inputs, get_text_from_logits, one_hot, decode_with_embedding

wandb.init(project='reverse decoding continuous prefix')

'''
Decoding to the lest, given right context. We want to find a left-prefix such that it leads to a certain generation on the right side.
'''

model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
device = 'cuda'
model.to(device)
model.eval()


def compute_continuous_prompts(desired_ending_ids, prefix_length, batch_size,
                               max_iter=200,
                               model_name=model,
                               lr=300.0,
                               step_size=50,
                               gamma=0.9):

    print(f"prefix_length: {prefix_length}, batch_size: {batch_size}, max_iter: {max_iter}")
    desired_ending_length = desired_ending_ids.size()[1]  # the length of the provided phrase
    assert desired_ending_length >= 1, "the provided sentence is a bit too short . . .  "
    assert desired_ending_length < 50, "the provided sentence is a bit too long . . .  "

    if True:
        optimized_embeddings = torch.nn.Parameter(
            torch.rand([batch_size, prefix_length, model_name.config.n_embd], device=device))
    else:
        perfect_prompt_ids = tokenizer.encode("The dog", return_tensors="pt").to(device)
        inputs_embeds = model_name.transformer.wte(perfect_prompt_ids)
        optimized_embeddings = torch.nn.Parameter(inputs_embeds.repeat(batch_size, 1, 1)).to(device)

    optimizer = torch.optim.Adam([optimized_embeddings], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size)
    temperature = 0.01
    # length = prefix_length + desired_ending_length

    for iter in range(max_iter):
        past = None
        inputs_embeds = None
        logits_so_far = None
        for i in range(desired_ending_length):
            if past is None:
                inputs_embeds = optimized_embeddings
            model_outputs = model_name(past_key_values=past, inputs_embeds=inputs_embeds)
            logits = model_outputs.logits
            past = model_outputs.past_key_values
            logits = logits[:, -1, :]
            logits = logits.unsqueeze(1)
            logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)

            # with straight-through
            if True:
                logits_so_far = (logits_so_far.detach() / temperature - logits_so_far).detach() + logits_so_far
                inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device=device)
            else:
                inputs_embeds = embed_inputs(model_name.get_input_embeddings(), logits / temperature, device=device)

        if iter % 100 == 99:
            print(" - - - - ")
            predicted_logits = decode_with_embedding(model_name, desired_ending_length, temperature, device,
                                                     optimized_embeddings)
            for batch_id in range(batch_size):
                predicted, nll, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
                print(
                    f" * batch ({batch_id}/{batch_size}) - iter: {iter}: prefix (len: {prefix_length}) ---> prediction: {predicted} (len: {desired_ending_length})")
                text, nll, _ = get_text_from_logits(predicted_logits[batch_id, :, :], tokenizer)
                print(
                    f" * batch ({batch_id}/{batch_size}) - iter: {iter}: model output (prompted with dense prompt): {text}")

        # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
        # compute loss with respect to the ending
        right_context_probability = nn.CrossEntropyLoss()(logits_so_far.view(-1, logits_so_far.size(-1)),
                                                          desired_ending_ids.view(-1).repeat(batch_size))
        _loss = right_context_probability
        _loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
        optimizer.step()
        scheduler.step()

        grad_norms = [p.grad.data.norm(2).tolist() for p in
                      list(filter(lambda p: p.grad is not None, model_name.parameters()))]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

        output = {
            "total_loss": _loss.detach().tolist(),
            "right_context_probability": right_context_probability.detach().tolist(),
            "right_context_probability_log": torch.log(right_context_probability).detach().tolist(),
            'avg_grad_norm_log': math.log(avg_grad_norm),
            'lr': scheduler.get_last_lr()[0]
        }

        # measuring the diversity of the prompts
        if False and iter % 10 == 0:
            cosine_distances = []
            # average distance between any pair of embeddijngs
            for iter1 in range(batch_size):
                for iter2 in range(batch_size):
                    if iter1 == iter2:
                        continue
                v1 = optimized_embeddings[iter1, :, :]
                v2 = optimized_embeddings[iter2, :, :]
                dist = torch.nn.CosineSimilarity()(v1, v2)
                cosine_distances.append(sum(dist.tolist()) / len(dist.tolist()))
            long_pairs_ratio = [1.0 if dist > 0.2 else 0.0 for dist in cosine_distances]
            output['overall_cosine_dist'] = sum(cosine_distances) / len(cosine_distances)
            output['long_pairs_ratio'] = sum(long_pairs_ratio) / len(long_pairs_ratio)

        wandb.log(output)

        optimizer.zero_grad()

    # torch.save(optimized_embeddings.data, f'optimized_prompts/optimized_prompt_{desired_ending.replace(".", "").replace(" ", "_")}.pt')
    return optimized_embeddings


def experiment1():
    desired_ending = "jumped to bite."
    desired_ending_ids = tokenizer.encode(desired_ending, return_tensors="pt").to(device)
    # assert input_ids.size() == 2, f"sizes don't match {input_ids.size()} ({input_ids}) vs 2"

    # embeddings of a prefix: [num-batches x num-tokens x VOCAB]
    compute_continuous_prompts(desired_ending_ids, prefix_length=2, batch_size=20, max_iter=2000, lr=300.0,
                               step_size=50)


def experiment2():
    '''
    how many of the prompts overlaps with the gold prompt?
    '''
    batch_size = 20
    desired_ending = "jumped to bite."
    desired_ending_ids = tokenizer.encode(desired_ending, return_tensors="pt").to(device)
    # assert input_ids.size() == 2, f"sizes don't match {input_ids.size()} ({input_ids}) vs 2"

    desired_prompt = 'The dog'
    desired_prompt_ids = tokenizer.encode(desired_prompt, return_tensors="pt").to(device)
    prefix_length = desired_prompt_ids.size()[1]
    desired_prompt_one_hot = one_hot(desired_prompt_ids, dimension=tokenizer.vocab_size)
    desired_prompt_embedding = torch.matmul(desired_prompt_one_hot.type(torch.FloatTensor).to(device),
                                            model.get_input_embeddings().weight)

    # embeddings of a prefix: [num-batches x num-tokens x VOCAB]
    optimized_embeddings = compute_continuous_prompts(desired_ending_ids, prefix_length=prefix_length,
                                                      batch_size=batch_size, max_iter=400)

    cosine_distances = []
    cosine_distances_to_gold_prompt = []
    # average distance between any pair of embeddings
    for iter1 in range(batch_size):
        v1 = optimized_embeddings[iter1, :, :]
        dist = torch.nn.CosineSimilarity()(v1, desired_prompt_embedding[0, :])
        cosine_distances_to_gold_prompt.append(sum(dist.tolist()) / len(dist.tolist()))
        for iter2 in range(batch_size):
            if iter1 == iter2:
                continue

            v2 = optimized_embeddings[iter2, :, :]
            dist = torch.nn.CosineSimilarity()(v1, v2)
            cosine_distances.append(sum(dist.tolist()) / len(dist.tolist()))
        # long_pairs_ratio = [1.0 if dist > 0.2 else 0.0 for dist in cosine_distances]

    print(cosine_distances_to_gold_prompt)
    count_nearby = len([x for x in cosine_distances_to_gold_prompt if abs(x) > 0.95])
    print(f" * count_nearby: {count_nearby}")
    print(f" * cosine_distances_to_gold_prompt: {len(cosine_distances_to_gold_prompt)}")


experiment1()
# experiment2()
