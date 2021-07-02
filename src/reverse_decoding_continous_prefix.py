from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from torch import nn
import wandb

wandb.init(project='reverse decoding')


def embed_inputs(embedding, logits, device='cuda'):
    '''
    embeds inputs in a dense representation, before passing them to the model
    '''
    probs = F.softmax(logits, dim=-1)
    probs = probs.to(device)
    return torch.matmul(probs, embedding.weight)


def _greedy(logits):
    _, last = torch.topk(logits, k=1, dim=-1)
    return last


def get_text_from_logits(logits, tokenizer):
    output_so_far = None
    last = None
    logp = 0
    for i in range(logits.shape[0]):
        last = _greedy(logits[i, :])
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=0)
        logp += logits[i, :].log_softmax(-1)[last.item()].item()
    nll = -logp
    text = tokenizer.decode(output_so_far.tolist())
    text = text.replace('\n', ' ')
    return text, nll, output_so_far


model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to('cuda')
model.eval()

input_ids = tokenizer.encode(" was found in a field.", return_tensors="pt").to('cuda')
# assert input_ids.size() == 2, f"sizes don't match {input_ids.size()} ({input_ids}) vs 2"
phrase_length = input_ids.size()[1]  # the length of the provided phrase
assert phrase_length > 2, "the provided sentence is a bit too short . . .  "
assert phrase_length < 20, "the provided sentence is a bit too long . . .  "

# embeddings of a prefix: [num-batches x num-tokens x VOCAB]
prefix_length = 1
batch_size = 1
if True:
    optimized_embeddings = torch.nn.Parameter(
        torch.rand([batch_size, prefix_length, model.config.n_embd], device='cuda'))
else:
    perfect_prompt_ids = tokenizer.encode("The dog", return_tensors="pt").to('cuda')
    inputs_embeds = model.transformer.wte(perfect_prompt_ids)
    optimized_embeddings = torch.nn.Parameter(inputs_embeds.repeat(batch_size, 1, 1)).to('cuda')

lr = 0.001
step_size = 1
optim = torch.optim.Adam([optimized_embeddings], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size)
temperature = 0.01
length = prefix_length + phrase_length

w = 1.0
for iter in range(1000):
    past = None
    inputs_embeds = None
    logits_so_far = None
    for i in range(phrase_length):
        if past is None:
            inputs_embeds = optimized_embeddings
        model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
        logits = model_outputs.logits
        past = model_outputs.past_key_values
        logits = logits[:, -1, :] / temperature
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device='cuda')

    probs_so_far = F.softmax(logits_so_far, dim=2)

    # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
    # compute loss with respect to the ending
    right_context_probability = nn.CrossEntropyLoss()(probs_so_far.view(-1, probs_so_far.size(-1)),
                                                      input_ids.view(-1).repeat(batch_size))
    _loss = right_context_probability
    _loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
    optim.step()
    scheduler.step()

    print(" - - - - ")
    for batch_id in range(batch_size):
        predicted, nll, _ = get_text_from_logits(logits_so_far[batch_id, :, :], tokenizer)
        print(f" * prefix: <--> {predicted}")

    grad_norms = [p.grad.data.norm(2).tolist() for p in list(filter(lambda p: p.grad is not None, model.parameters()))]
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

    wandb.log({
        "total_loss": _loss.detach().tolist(),
        "_dist": right_context_probability.detach().tolist(),
        'avg_grad_norm': avg_grad_norm,
    })

    model.zero_grad()
