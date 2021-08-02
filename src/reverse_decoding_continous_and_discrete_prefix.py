from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import nn
from utils import embed_inputs, get_text_from_logits, svd_model_embeddings
import torch
import wandb
import math
import torch.nn.functional as F

def optimize_logits_and_embeddings_jointly(desired_ending_ids, prefix_length, batch_size):
    '''
    Decoding to the lest, given right context. We want to find a left-prefix such that it leads to a certain generation on the right side.
    '''
    wandb.init(project='reverse decoding continuous and discrete prefix')

    # assert input_ids.size() == 2, f"sizes don't match {input_ids.size()} ({input_ids}) vs 2"
    desired_ending_length = desired_ending_ids.size()[1]  # the length of the provided phrase
    assert desired_ending_length >= 1, "the provided sentence is a bit too short . . .  "
    assert desired_ending_length < 20, "the provided sentence is a bit too long . . .  "

    # embeddings of a prefix: [num-batches x num-tokens x VOCAB]
    if True:
        optimized_embeddings = torch.nn.Parameter(
            torch.rand([batch_size, prefix_length, model.config.n_embd], device=device))
        optimized_word_logits = torch.nn.Parameter(
            torch.rand([batch_size, prefix_length, tokenizer.vocab_size], device='cuda')
        )
    else:
        perfect_prompt_ids = tokenizer.encode("The dog", return_tensors="pt").to(device)
        inputs_embeds = model.transformer.wte(perfect_prompt_ids)
        optimized_embeddings = torch.nn.Parameter(inputs_embeds.repeat(batch_size, 1, 1)).to(device)

    w = 0.5
    lr = 2.0
    step_size = 20
    optimizer = torch.optim.Adam([optimized_embeddings, optimized_word_logits], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.99)
    temperature = 0.01
    # dynamic_temperature = 1000
    length = prefix_length + desired_ending_length

    for iter in range(2000):
        norm = torch.nn.L1Loss()
        # norm = torch.nn.MSELoss()
        optimized_word_probs = torch.nn.Softmax(dim=2)(optimized_word_logits / temperature)
        optimized_word_probs_no_temp = torch.nn.Softmax(dim=2)(optimized_word_logits)
        # with straight-through
        if True:
            optimized_word_probs = (optimized_word_probs.detach() - optimized_word_probs_no_temp).detach() + optimized_word_probs_no_temp
        # optimized_word_probs = torch.abs(optimized_word_logits) / torch.sum(optimized_word_logits)
        # optimized_word_probs = optimized_word_logits
        embedding_loss = norm(optimized_embeddings, torch.matmul(optimized_word_probs, model.get_input_embeddings().weight))

        # norm_loss = norm(torch.ones([batch_size, prefix_length]).to(device), torch.sum(optimized_word_probs, dim=2))

        # entropy = torch.mean(
        #     # entropy for each position
        #     torch.sum(-torch.log(optimized_word_probs + 0.000001) * optimized_word_probs, dim=2)
        # )
        entropy = torch.FloatTensor(0)

        # if w > 0.5 and iter % 5 == 0:
        #     w -= 0.02

        # if iter % 2 == 0:
        #     dynamic_temperature *= 0.97

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
        right_context_probability = nn.CrossEntropyLoss()(logits_so_far.view(-1, logits_so_far.size(-1)),
                                                          desired_ending_ids.view(-1).repeat(batch_size))

        _loss = w * right_context_probability + (1-w) * (embedding_loss)
        _loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
        optimizer.step()
        scheduler.step()

        if iter % 10 == 0:
            print(" - - - - ")
            for batch_id in range(batch_size):
                predicted, _, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
                optimized_prefix, _, _ = get_text_from_logits(optimized_word_logits[batch_id, :, :], tokenizer)
                print(f" * prefix: {optimized_prefix} ---> prediction: {predicted}")
                # print(f" * temperature: {dynamic_temperature}")
                # print(f" * w: {w}")
            print(f" * loss: {_loss}")

        grad_norms = [p.grad.data.norm(2).tolist() for p in
                      list(filter(lambda p: p.grad is not None, model.parameters()))]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

        wandb.log({
            "total_loss": _loss.detach().tolist(),
            "total_loss_log": torch.log(_loss).detach().tolist(),
            "right_context_probability": right_context_probability.detach().tolist(),
            "right_context_probability_log": torch.log(right_context_probability).detach().tolist(),
            'embedding_loss': embedding_loss.detach().tolist(),
            'embedding_loss_log': torch.log(embedding_loss).detach().tolist(),
            'avg_grad_norm_log': math.log(avg_grad_norm),
            'lr': scheduler.get_last_lr()[0],
            'entropy': entropy.detach().tolist(),
        })

        optimizer.zero_grad()

    # torch.save(optimized_embeddings.data, f'optimized_prompts/optimized_prompt_{desired_ending.replace(".", "").replace(" ", "_")}.pt')

def optimize_emeddings_decomposed_projections(desired_ending_ids, prefix_length, batch_size, max_iter=200):
    '''
    first, decompose the embedding matrix into orthogonal basis
    then optimize an embedding vector with respect to these orthogonal basis
    '''
    wandb.init(project='reverse decoding continuous and discrete prefix, with svd')
    desired_ending_length = desired_ending_ids.size()[1]  # the length of the provided phrase
    assert desired_ending_length >= 1, "the provided sentence is a bit too short . . .  "
    assert desired_ending_length < 50, "the provided sentence is a bit too long . . .  "

    if True:
        optimized_embeddings = torch.nn.Parameter(
            torch.rand([batch_size, prefix_length, model.config.n_embd], device=device))
    else:
        perfect_prompt_ids = tokenizer.encode("The dog", return_tensors="pt").to(device)
        inputs_embeds = model.transformer.wte(perfect_prompt_ids)
        optimized_embeddings = torch.nn.Parameter(inputs_embeds.repeat(batch_size, 1, 1)).to(device)

    with torch.no_grad():
        u, s, vh = svd_model_embeddings(model)

    lr = 10.0
    step_size = 500
    optimizer = torch.optim.Adam([optimized_embeddings], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.9)
    temperature = 0.01
    w = 0.5
    # length = prefix_length + desired_ending_length

    for iter in range(max_iter):
        past = None
        inputs_embeds = None
        logits_so_far = None
        for i in range(desired_ending_length):
            if past is None:
                inputs_embeds = torch.matmul(optimized_embeddings, s * vh)
            model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
            logits = model_outputs.logits
            past = model_outputs.past_key_values
            logits = logits[:, -1, :]
            logits = logits.unsqueeze(1)
            logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)

            # with straight-through
            if True:
                logits_so_far = (logits_so_far.detach() / temperature - logits_so_far).detach() + logits_so_far
            inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device=device)

        # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
        # compute loss with respect to the ending
        right_context_probability = nn.CrossEntropyLoss()(logits_so_far.view(-1, logits_so_far.size(-1)),
                                                          desired_ending_ids.view(-1).repeat(batch_size))

        # cosine similarity between the basis vectors and the optimized embeddings
        # similarity = torch.abs(torch.matmul(optimized_embeddings, torch.transpose(u_subset, 0, 1))) # / torch.norm(optimized_embeddings, p=2)
        # similarity = torch.matmul(optimized_embeddings.squeeze(0), torch.transpose(u_subset, 0, 1)) # / torch.norm(optimized_embeddings, p=2))
        # similarity = torch.nn.CosineSimilarity(dim=1)(optimized_embeddings.squeeze(0), u)
        # normalized_embeddings = optimized_embeddings / torch.norm(optimized_embeddings, p=2, dim=2).unsqueeze(2)
        # similarity = torch.matmul(normalized_embeddings, torch.transpose(u, 0, 1))
        v1 = optimized_embeddings.unsqueeze(1)
        v2 = u.unsqueeze(0).unsqueeze(2)
        similarity = torch.nn.CosineSimilarity(dim=3)(v1, v2)
        mean_sim_to_basis = torch.mean(similarity)
        max_sim_to_basis = torch.mean(torch.max(similarity, dim=1).values)
        # max_sim_to_basis = torch.max(similarity)
        _loss = w * right_context_probability - (1 - w) * max_sim_to_basis

        # print(" - - - - - ")
        # print(torch.cuda.memory_summary(abbreviated=True))
        # print(torch.cuda.memory_allocated())
        _loss.backward(retain_graph=True)
        # print(torch.cuda.memory_summary(abbreviated=True))
        # print(torch.cuda.memory_allocated())

        # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
        optimizer.step()
        scheduler.step()

        if iter % 50 == 0:
            print(" - - - - ")
            for batch_id in range(batch_size):
                predicted, nll, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
                # most_similar = torch.argmax(similarity[batch_id, :, :],dim=1)
                # prompt_text = tokenizer.decode(most_similar.tolist())
                # print(f" * prefix: {prompt_text} (len: {prefix_length}) ---> prediction: {predicted} (len: {desired_ending_length})")
                print(f" * prefix: (len: {prefix_length}) ---> prediction: {predicted} (len: {desired_ending_length})")

        #
        # grad_norms = [p.grad.data.norm(2).tolist() for p in
        #               list(filter(lambda p: p.grad is not None, model.parameters()))]
        # avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

        output = {
            "total_loss": _loss.detach().tolist(),
            "right_context_probability": right_context_probability.detach().tolist(),
            "right_context_probability_log": torch.log(right_context_probability).detach().tolist(),
            # 'avg_grad_norm_log': math.log(avg_grad_norm),
            'lr': scheduler.get_last_lr()[0],
            'mean_sim_to_basis': mean_sim_to_basis.detach().tolist(),
            'max_sim_to_basis': max_sim_to_basis.detach().tolist(),
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



device = 'cuda'
model_size = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to(device)
model.eval()

def experiment1():
    desired_ending = "jumped to bite."
    desired_ending_ids = tokenizer.encode(desired_ending, return_tensors="pt").to(device)
    prefix_length = 2
    batch_size = 1
    optimize_logits_and_embeddings_jointly(desired_ending_ids, prefix_length, batch_size)


def experiment2():
    desired_ending = "jumped to bite."
    desired_ending_ids = tokenizer.encode(desired_ending, return_tensors="pt").to(device)
    prefix_length = 2
    batch_size = 3
    optimize_emeddings_decomposed_projections(desired_ending_ids, prefix_length, batch_size, max_iter=10000)

experiment2()
