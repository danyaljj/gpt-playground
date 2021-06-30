from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from torch import nn
import wandb
from torch.distributions.utils import probs_to_logits, logits_to_probs

wandb.init(project='reverse decoding')


def one_hot(tensor, dimension):
    while len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    onehot = torch.LongTensor(tensor.shape[0], tensor.shape[1], dimension).to(tensor.device)
    onehot.zero_().scatter_(2, tensor.unsqueeze(-1), 1)
    return onehot


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
# input_ids_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
# assert input_ids.size() == 2, f"sizes don't match {input_ids.size()} ({input_ids}) vs 2"
phrase_length = input_ids.size()[1]  # the length of the provided phrase
assert phrase_length > 2, "the provided sentence is a bit too short . . .  "
assert phrase_length < 20, "the provided sentence is a bit too long . . .  "

# the logits of the word that we want to predict
# since it's a single token, the size is [num-batches x num-tokens x VOCAB]
prefix_length = 2
# uniform distribution over all the vocab
if False:
    optimized_logits = torch.nn.Parameter(
        -torch.log(torch.rand([1, prefix_length, tokenizer.vocab_size], device='cuda') / tokenizer.vocab_size)
    )
else:
    optimized_logits = torch.ones([1, prefix_length, tokenizer.vocab_size], device='cuda') * -10.0
    perfect_prompt = tokenizer.encode("The dog", return_tensors="pt")
    for position_id, word_id in enumerate(perfect_prompt[0]):
        optimized_logits[0, position_id, word_id] = 0
    print(torch.sum(optimized_logits))
    optimized_logits = torch.nn.Parameter(optimized_logits)

lr = 0.05
step_size = 20
optim = torch.optim.Adam([optimized_logits], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size)
temperature = 0.01
length = prefix_length + phrase_length

# TODO:
#  - start from the perfect prompt
#  - the issue might be that GPT2 is trained on discrete prompts
#       - in which case, maybe I should try argmax instead of soft distribution
#  - add a regularization term (entropy, instead of temperature) to force the logits towards more one-hot


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


    probs_so_far = F.softmax(logits_so_far, dim=2)

    # compute loss with respect to the ending
    # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
    _dist = nn.CrossEntropyLoss()(probs_so_far[0, :, :], input_ids[0, :])

    optimized_probs = F.softmax(optimized_logits, dim=2) # TODO: replace it with logits_to_probs
    # minimize entropy so that we get peaky distributions
    _entropy = -torch.mean(
        # entropy for each position
        torch.sum(torch.log(optimized_probs) * optimized_probs, dim=2)
    )

    # _sum1_loss = torch.norm(torch.sum(optimized_probs, dim=2) - 1.0)
    # _sum1_loss = torch.nn.MSELoss()(torch.exp(optimized_logits), torch.ones(prefix_length))

    w = 0.9
    _loss = (1 - w) * _entropy + w * _dist  # + _sum1_loss # _dist + +
    _sum = torch.sum(optimized_probs)

    _loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
    optim.step()
    scheduler.step()

    # projection step: make the logits more peakier
    with torch.no_grad():
        # optimized_probs = F.softmax(100 * optimized_probs, dim=2)
        # optimized_logits = logits_to_probs(optimized_probs)
        optimized_logits += 0.01 * torch.randn(optimized_logits.size()).to('cuda')

    _entropy2 = -torch.mean(
        # entropy for each position
        torch.sum(torch.log(optimized_probs) * optimized_probs, dim=2)
    )

    text, nll, _ = get_text_from_logits(optimized_logits[0, :, :], tokenizer)
    print(f" * prefix: {text}")

    text, nll, _ = get_text_from_logits(logits_so_far[0, :, :], tokenizer)
    print(f" * predicted: {text}")

    grad_norms = [p.grad.data.norm(2).tolist() for p in list(filter(lambda p: p.grad is not None, model.parameters()))]
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

    wandb.log({
        "total_loss": _loss.detach().tolist(),
        "_dist": _dist.detach().tolist(),
        "sum": _sum.detach().tolist(),
        '_entropy': _entropy.detach().tolist(),
        # '_entropy2': _entropy2.detach().tolist(),
        'avg_grad_norm': avg_grad_norm,
        # '_sum1_loss': _sum1_loss
    })

    # break
    model.zero_grad()
