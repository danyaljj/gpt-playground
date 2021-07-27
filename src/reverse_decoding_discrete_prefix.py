from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from torch import nn
import wandb
from utils import embed_inputs, get_text_from_logits

wandb.init(project='reverse decoding')

torch.random.initial_seed()

'''
Decoding to the lest, given right context. We want to find a left-tokens such that it leads to a certain generation on the right side.
'''

device = 'cuda'
model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to(device)
model.eval()

right_context_ids = tokenizer.encode("barked at the neighbor.", return_tensors="pt").to(device)
# input_ids_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
phrase_length = right_context_ids.size()[1]  # the length of the provided phrase
assert phrase_length >= 1, "the provided sentence is a bit too short . . .  "
assert phrase_length < 20, "the provided sentence is a bit too long . . .  "

# the logits of the word that we want to predict
# since it's a single token, the size is [num-batches x num-tokens x VOCAB]
prefix_length = 2
batch_size = 1

if False:
    # uniform distribution over all the vocab
    optimized_logits = torch.nn.Parameter(
        torch.randn([batch_size, prefix_length, tokenizer.vocab_size], device=device)
    )
else:
    optimized_logits = torch.ones([batch_size, prefix_length, tokenizer.vocab_size], device=device) * -10.0
    perfect_prompt = tokenizer.encode("The dog", return_tensors="pt")
    for position_id, word_id in enumerate(perfect_prompt[0]):
        optimized_logits[:, position_id, word_id] = 0
    optimized_logits = torch.nn.Parameter(optimized_logits)

lr = 100000
step_size = 1000
optimizer = torch.optim.Adam([optimized_logits], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.9)
dynamic_temperature = 1000
length = prefix_length + phrase_length

for iter in range(1000):
    optimized_probs = torch.nn.Softmax(dim=2)(optimized_logits / dynamic_temperature)
    _entropy = -torch.mean(
        # entropy for each position
        torch.sum(torch.log(optimized_probs + 0.000001) * optimized_probs, dim=2)
    )

    # iter % 5 == 0 and
    if _entropy.detach().tolist() > 6:
        dynamic_temperature *= 0.9999

    temperature = 0.01
    past = None
    inputs_embeds = None
    logits_so_far = None
    for i in range(phrase_length):
        if past is None:
            # the embeddings extracted from the optimized parameters
            inputs_embeds = embed_inputs(model.get_input_embeddings(), optimized_probs, device=device)
        model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
        logits = model_outputs.logits
        past = model_outputs.past_key_values
        logits = logits[:, -1, :]
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        inputs_embeds = embed_inputs(model.get_input_embeddings(), logits / temperature, device=device)

        # if i < phrase_length:
        if i + 1 == phrase_length:
            # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
            # compute loss with respect to the ending
            _right_context_probs = nn.CrossEntropyLoss()(logits_so_far.view(-1, logits_so_far.size(-1)),
                                                         right_context_ids.view(-1)[0:i+1].repeat(batch_size))

            # _sum1_loss = torch.norm(torch.sum(optimized_probs, dim=2) - 1.0)
            # _sum1_loss = torch.nn.MSELoss()(torch.exp(optimized_logits), torch.ones(prefix_length))

            w = 1.0
            # _loss = (1 - w) * _entropy + w * _right_context_probs
            _loss = _right_context_probs
            _loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
            optimizer.step()
            scheduler.step()

            # projection step: make the logits more peakier
            # if iter % 50 == 0:
            #     with torch.no_grad():
            #         # optimized_probs = F.softmax(100 * optimized_probs, dim=2)
            #         # optimized_logits = logits_to_probs(optimized_probs)
            #         optimized_logits += 0.01 * torch.randn(optimized_logits.size()).to('cuda')

            print(" - - - - ")
            print(f" * temperature: {dynamic_temperature}")
            for batch_id in range(batch_size):
                prefix, nll, _ = get_text_from_logits(optimized_logits[batch_id, :, :], tokenizer)
                predicted, nll, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
                print(f" * prefix: {prefix} <--> generated text: {predicted}")

            grad_norms = [p.grad.data.norm(2).tolist() for p in list(filter(lambda p: p.grad is not None, model.parameters()))]
            avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

            _sum = torch.sum(optimized_probs)
            wandb.log({
                "total_loss": _loss.detach().tolist(),
                "_right_context_probs": _right_context_probs.detach().tolist(),
                "sum": _sum.detach().tolist(),
                '_entropy': _entropy.detach().tolist(),
                # '_entropy2': _entropy2.detach().tolist(),
                'avg_grad_norm': avg_grad_norm,
                # '_sum1_loss': _sum1_loss
            })

            optimizer.zero_grad()
