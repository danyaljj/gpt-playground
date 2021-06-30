from torch.distributions import Categorical
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F


def embed_inputs(embedding, logits, device='cuda', print_entropy=False):
    '''
    embeds inputs in a dense representation, before passing them to the model
    '''
    # typically we embed a one-hot vector. But here since we work we work with dense representations,
    # we have softmax here to make sure that all the values of the input logits sum to one (similar to a 1-hot vector).
    probs = F.softmax(logits, dim=-1)
    # probs = logits
    if print_entropy:
        _entropy = - probs * torch.log(probs)
        _entropy = torch.sum(_entropy)
        print(_entropy)

    probs = probs.to(device)
    return torch.matmul(probs, embedding.weight)


def _greedy(logits):
    _, last = torch.topk(logits, k=1, dim=-1)
    return last


def one_hot(tensor, dimension):
    while len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    onehot = torch.LongTensor(tensor.shape[0], tensor.shape[1], dimension).to(tensor.device)
    onehot.zero_().scatter_(2, tensor.unsqueeze(-1), 1)
    return onehot


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
input_ids = tokenizer.encode("The dog", return_tensors="pt").to('cuda')
input_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)


def decode(model, length, temperature, device):
    '''
    GPT2 decoding via dense representations (no arg-max)
    '''
    past = None
    inputs_embeds = None
    logits_so_far = None
    for i in range(length):
        if past is None:
            # inputs_embeds = model.transformer.wte(input_ids)
            inputs_embeds = embed_inputs(model.get_input_embeddings(), input_one_hot.type(torch.FloatTensor)/ temperature, device='cuda', print_entropy=True)
        model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
        logits = model_outputs.logits
        past = model_outputs.past_key_values
        logits = logits[:, -1, :] / temperature
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device=device)
    return logits_so_far


def query_via_embeddings():
    input_ids = tokenizer("In my early life", return_tensors='pt')['input_ids'].to('cuda')
    inputs_embeds = model.transformer.wte(input_ids).squeeze()

    out1 = model(inputs_embeds=inputs_embeds)
    out2 = model(input_ids=input_ids)

    input_ids_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
    input_embeddings2 = embed_inputs(model.get_input_embeddings(), input_ids_one_hot.type(torch.FloatTensor),device='cuda')
    out3 = model(inputs_embeds=input_embeddings2)
    print(out1)
    print(out2)


for temperature in [0.001, 0.01, 0.1, 0.2, 0.3, 1, 5]:
    print(f" ------- \n * temperature: {temperature}")
    logits_so_far = decode(model, 100, temperature, 'cuda')
    text, nll, _ = get_text_from_logits(logits_so_far[0, :, :], tokenizer)
    print(text)
