from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch import nn
import wandb
import math
from utils import embed_inputs, get_text_from_logits

wandb.init(project='reverse decoding continuous and discrete prefix')

'''
Decoding to the lest, given right context. We want to find a left-prefix such that it leads to a certain generation on the right side.
'''

model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
device = 'cuda'
model.to(device)
model.eval()

desired_ending = "jumped to bite."
desired_ending_ids = tokenizer.encode(desired_ending, return_tensors="pt").to(device)
# assert input_ids.size() == 2, f"sizes don't match {input_ids.size()} ({input_ids}) vs 2"
desired_ending_length = desired_ending_ids.size()[1]  # the length of the provided phrase
assert desired_ending_length >= 1, "the provided sentence is a bit too short . . .  "
assert desired_ending_length < 20, "the provided sentence is a bit too long . . .  "

# embeddings of a prefix: [num-batches x num-tokens x VOCAB]
prefix_length = 2
batch_size = 1
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
lr = 10.0
step_size = 200
optimizer = torch.optim.Adam([optimized_embeddings, optimized_word_logits], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.85)
temperature = 0.01
# dynamic_temperature = 1000
length = prefix_length + desired_ending_length

for iter in range(2000):
    norm = torch.nn.L1Loss()
    # optimized_word_probs = torch.nn.Softmax(dim=2)(optimized_word_logits / temperature)
    # with straight-through
    # if True:
    #     optimized_word_probs = (optimized_word_probs.detach() - optimized_word_logits).detach() + optimized_word_logits
    # optimized_word_probs = torch.abs(optimized_word_logits) / torch.sum(optimized_word_logits)
    optimized_word_probs = optimized_word_logits
    embedding_loss = norm(optimized_embeddings, torch.matmul(optimized_word_probs, model.get_input_embeddings().weight))

    norm_loss = norm(torch.ones([batch_size, prefix_length]).to(device), torch.sum(optimized_word_probs, dim=2))

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

    _loss = w * right_context_probability + (1-w) * (norm_loss + embedding_loss)
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
        'avg_grad_norm_log': math.log(avg_grad_norm),
        'lr': scheduler.get_last_lr()[0],
        'embedding_loss': embedding_loss.detach().tolist(),
        'entropy': entropy.detach().tolist(),
    })

    optimizer.zero_grad()

torch.save(optimized_embeddings.data, f'optimized_prompts/optimized_prompt_{desired_ending.replace(".", "").replace(" ", "_")}.pt')
