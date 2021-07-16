from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from torch import nn
import wandb
from utils import embed_inputs, get_text_from_logits

wandb.init(project='reverse decoding')

'''
Decoding to the lest, given right context. We want to find a left-tokens such that it leads to a certain generation on the right side.
'''

model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to('cuda')
model.eval()

right_context_ids = tokenizer.encode("good", return_tensors="pt").to('cuda')
# input_ids_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
phrase_length = right_context_ids.size()[1]  # the length of the provided phrase
assert phrase_length >= 1, "the provided sentence is a bit too short . . .  "
assert phrase_length < 20, "the provided sentence is a bit too long . . .  "

# the logits of the word that we want to predict
# since it's a single token, the size is [num-batches x num-tokens x VOCAB]
prefix_length = 2
batch_size = 8
if True:
    # uniform distribution over all the vocab
    optimized_logits = torch.nn.Parameter(
        torch.rand([batch_size, prefix_length, tokenizer.vocab_size], device='cuda')
    )
else:
    optimized_logits = torch.ones([batch_size, prefix_length, tokenizer.vocab_size], device='cuda') * -10.0
    perfect_prompt = tokenizer.encode("The dog", return_tensors="pt")
    for position_id, word_id in enumerate(perfect_prompt[0]):
        optimized_logits[:, position_id, word_id] = 0
    optimized_logits = torch.nn.Parameter(optimized_logits)

lr = 0.00001
step_size = 1  # TODO: need to play with this
optimizer = torch.optim.Adam([optimized_logits], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size)
temperature = 0.01
length = prefix_length + phrase_length

for iter in range(1000):
    past = None
    inputs_embeds = None
    logits_so_far = None
    for i in range(phrase_length):
        if past is None:
            # the embeddings extracted from the optimized parameters
            inputs_embeds = embed_inputs(model.get_input_embeddings(), optimized_logits / temperature, device='cuda')
        model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
        logits = model_outputs.logits
        past = model_outputs.past_key_values
        logits = logits[:, -1, :] / temperature
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device='cuda')

    # convert logits to probabilities
    probs_so_far = F.softmax(logits_so_far, dim=2)
    optimized_probs = F.softmax(optimized_logits, dim=2)  # TODO: replace it with logits_to_probs

    # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
    # compute loss with respect to the ending
    _right_context_probs = nn.CrossEntropyLoss()(probs_so_far.view(-1, probs_so_far.size(-1)),
                                                 right_context_ids.view(-1).repeat(batch_size))

    # TODO: instead of maximizing entropy, maximize the maximum of the probabilty distribution
    # minimize entropy so that we get peaky distributions
    _entropy = -torch.mean(
        # entropy for each position
        torch.sum(torch.log(optimized_probs) * optimized_probs, dim=2)
    )

    # _sum1_loss = torch.norm(torch.sum(optimized_probs, dim=2) - 1.0)
    # _sum1_loss = torch.nn.MSELoss()(torch.exp(optimized_logits), torch.ones(prefix_length))

    w = 0.99
    _loss = (1 - w) * _entropy + w * _right_context_probs
    _sum = torch.sum(optimized_probs)

    _loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
    optimizer.step()
    scheduler.step()

    # projection step: make the logits more peakier
    if iter % 50 == 0:
        with torch.no_grad():
            # optimized_probs = F.softmax(100 * optimized_probs, dim=2)
            # optimized_logits = logits_to_probs(optimized_probs)
            optimized_logits += 0.01 * torch.randn(optimized_logits.size()).to('cuda')

    _entropy2 = -torch.mean(
        # entropy for each position
        torch.sum(torch.log(optimized_probs) * optimized_probs, dim=2)
    )

    print(" - - - - ")
    for batch_id in range(batch_size):
        prefix, nll, _ = get_text_from_logits(optimized_logits[batch_id, :, :], tokenizer)
        predicted, nll, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
        print(f" * prefix: {prefix} <--> {predicted}")

    grad_norms = [p.grad.data.norm(2).tolist() for p in list(filter(lambda p: p.grad is not None, model.parameters()))]
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

    wandb.log({
        "total_loss": _loss.detach().tolist(),
        "_dist": _right_context_probs.detach().tolist(),
        "sum": _sum.detach().tolist(),
        '_entropy': _entropy.detach().tolist(),
        # '_entropy2': _entropy2.detach().tolist(),
        'avg_grad_norm': avg_grad_norm,
        # '_sum1_loss': _sum1_loss
    })

    optimizer.zero_grad()
