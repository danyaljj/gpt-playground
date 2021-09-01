import torch
import torch.nn.functional as F
from torch import nn

# used to prevent numerical issues
eps = 1e-10

def decode_with_embedding(model, length, temperature, device, prompt_embedding):
    '''
    GPT2 decoding via dense representations (no arg-max)
    '''
    past = None
    inputs_embeds = None
    logits_so_far = None
    for i in range(length):
        if past is None:
            inputs_embeds = prompt_embedding
        model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
        logits = model_outputs.logits
        past = model_outputs.past_key_values
        logits = logits[:, -1, :] / temperature
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device=device)
    return logits_so_far


def decode_with_one_hot(model, length, input_one_hot1, temperature, device):
    '''
    GPT2 decoding via dense representations (no arg-max)
    '''
    past = None
    inputs_embeds1 = None
    logits_so_far = None
    for i in range(length):
        if past is None:
            # inputs_embeds = model.transformer.wte(input_ids)
            inputs_embeds1 = embed_inputs(model.get_input_embeddings(),
                                          input_one_hot1.type(torch.FloatTensor) / temperature, device='cuda',
                                          print_entropy=True)
        model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds1)
        logits = model_outputs.logits
        past = model_outputs.past_key_values
        logits = logits[:, -1, :] / temperature
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        inputs_embeds1 = embed_inputs(model.get_input_embeddings(), logits, device=device)
    return logits_so_far

def decode_with_argmax(model, length, input_ids1, device):
    '''
    GPT2 decoding via dense representations (no arg-max)
    '''
    past = None
    logits_so_far = None
    for i in range(length):
        # if past is None:
        # inputs_embeds = model.transformer.wte(input_ids1)
        # inputs_embeds = embed_inputs(model.get_input_embeddings(), input_one_hot.type(torchc)/ temperature, device='cuda', print_entropy=True)
        model_outputs = model(input_ids=input_ids1)
        # else:
        #     model_outputs = model(past_key_values=past, input_ids=input_ids1)
        logits = model_outputs.logits
        # past = model_outputs.past_key_values
        logits = logits[:, -1, :]
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        # inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device=device)
        next_token = torch.argmax(logits)
        input_ids1 = torch.cat([input_ids1, next_token.unsqueeze(0).unsqueeze(0)], 1)
    return logits_so_far


def embed_inputs(embedding, logits, device, print_entropy=False):
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


def one_hot(tensor, dimension):
    while len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    onehot = torch.LongTensor(tensor.shape[0], tensor.shape[1], dimension).to(tensor.device)
    onehot.zero_().scatter_(2, tensor.unsqueeze(-1), 1)
    return onehot

def plot_histogram(x):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.hist(x)
    plt.show()


def svd_model_embeddings(model):
    '''
    uses SVD to decompose
    '''
    E = model.get_input_embeddings().weight
    # since the embedding matrix is not degenerate, `full_matrices` should not have any effect in the size of the matrices
    u, s, vh = torch.linalg.svd(E, full_matrices=False)

    # verify that this is a good decomposition
    r = torch.matmul(u, s * vh)
    diff = torch.mean(torch.abs(r - E))
    assert diff < 0.07, f"the diff is larger than expected {diff}"

    return u, s, vh

def project_ids(input_ids, model, tokenizer, device):
    desired_beginning_ids = tokenizer.encode(input_ids, return_tensors="pt").to(device)
    desired_beginning_one_hot = one_hot(desired_beginning_ids, dimension=tokenizer.vocab_size)
    embeddings = torch.matmul(desired_beginning_one_hot.type(torch.FloatTensor).to(device), model.get_input_embeddings().weight.to(device))
    return embeddings

def project_embeddings(embedding, model, temp, with_streight_through=False):
    logits = torch.matmul(embedding, torch.transpose(model.get_input_embeddings().weight, 0, 1))
    # if with_streight_through:
    #     logits = (logits.detach() / temp - logits).detach() + logits
    # else:
    #     logits = logits / temp
    probs = F.softmax(logits, dim=-1)
    # if with_streight_through:
    #     probs = (probs.detach() - logits).detach() + logits

    return probs