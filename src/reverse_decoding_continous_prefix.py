from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from torch import nn
import wandb
import math
from utils import embed_inputs, get_text_from_logits

wandb.init(project='reverse decoding continuous prefix')

'''
Decoding to the lest, given right context. We want to find a left-prefix such that it leads to a certain generation on the right side.
'''

model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to('cuda')
model.eval()

input_ids = tokenizer.encode("barked at the cat.", return_tensors="pt").to('cuda')
# assert input_ids.size() == 2, f"sizes don't match {input_ids.size()} ({input_ids}) vs 2"
phrase_length = input_ids.size()[1]  # the length of the provided phrase
assert phrase_length >= 1, "the provided sentence is a bit too short . . .  "
assert phrase_length < 20, "the provided sentence is a bit too long . . .  "

# embeddings of a prefix: [num-batches x num-tokens x VOCAB]
prefix_length = 2
batch_size = 1
if True:
    optimized_embeddings = torch.nn.Parameter(
        torch.rand([batch_size, prefix_length, model.config.n_embd], device='cuda'))
else:
    perfect_prompt_ids = tokenizer.encode("The dog", return_tensors="pt").to('cuda')
    inputs_embeds = model.transformer.wte(perfect_prompt_ids)
    optimized_embeddings = torch.nn.Parameter(inputs_embeds.repeat(batch_size, 1, 1)).to('cuda')

lr = 2.0
iters_per_token = 10
step_size = phrase_length * iters_per_token
optim = torch.optim.Adam([optimized_embeddings], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size, gamma=0.9)
temperature = 0.01
length = prefix_length + phrase_length

for _ in range(100):
    for intermediate_max in range(1, phrase_length):
        for iter in range(iters_per_token):
            past = None
            inputs_embeds = None
            logits_so_far = None
            for i in range(intermediate_max):
                if past is None:
                    inputs_embeds = optimized_embeddings
                model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
                logits = model_outputs.logits
                past = model_outputs.past_key_values
                logits = logits[:, -1, :]
                logits = logits.unsqueeze(1)
                logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
                inputs_embeds = embed_inputs(model.get_input_embeddings(), logits / temperature, device='cuda')

            probs_so_far = F.softmax(logits_so_far, dim=2)

            # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
            # compute loss with respect to the ending
            right_context_probability = nn.CrossEntropyLoss()(logits_so_far.view(-1, logits_so_far.size(-1)),
                                                              input_ids[:, :intermediate_max].view(-1).repeat(
                                                                  batch_size))
            _loss = right_context_probability
            _loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
            optim.step()
            scheduler.step()

            print(" - - - - ")
            for batch_id in range(batch_size):
                predicted, nll, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
                print(f" * prefix: <--> {predicted}")

            grad_norms = [p.grad.data.norm(2).tolist() for p in
                          list(filter(lambda p: p.grad is not None, model.parameters()))]
            avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

            wandb.log({
                "total_loss": _loss.detach().tolist(),
                "right_context_probability": right_context_probability.detach().tolist(),
                'avg_grad_norm_log': math.log(avg_grad_norm),
                'lr': scheduler.get_last_lr()[0],
            })

            model.zero_grad()
