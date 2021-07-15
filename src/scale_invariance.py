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


model1_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model1_size)

model1 = GPT2LMHeadModel.from_pretrained(model1_size, output_hidden_states=True)
model1.to('cuda')
model1.eval()

model2_size = "gpt2-medium"
model2 = GPT2LMHeadModel.from_pretrained(model2_size, output_hidden_states=True)
model2.to('cuda')
model2.eval()

input_ids = tokenizer.encode("In order to make an omelette", return_tensors="pt").to('cuda')
input_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)

# prepare the transformation matrices
model1_embedding_inv = torch.pinverse(model1.get_input_embeddings().weight)
transformation_1_to_2_matrix = torch.matmul(model1_embedding_inv, model2.get_input_embeddings().weight)

def decode(model, length, temperature, device, prompt_embedding):
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


def experiment1():
    '''
    In this experiment, we use the embedding matrices of the language models to convert encoded inputs of LM1 to encoded input of LM2
    '''
    prompt_embedding1 = torch.matmul(input_one_hot.type(torch.FloatTensor).to('cuda'), model1.get_input_embeddings().weight)
    prompt_embedding2 = torch.matmul(input_one_hot.type(torch.FloatTensor).to('cuda'), model2.get_input_embeddings().weight)

    print(" . . . ")
    transformed_prompt_embedding2 = torch.matmul(prompt_embedding1, transformation_1_to_2_matrix)

    # compare the embeddings
    diff = torch.abs(torch.mean(transformed_prompt_embedding2 - prompt_embedding2)).tolist()
    assert diff < 0.01, f'Diff: {diff} - the projection did not work! :-/ '

    temperature = 0.01 # to simulate the effect of arg-max
    logits = decode(model1, 50, temperature, 'cuda', prompt_embedding1)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 1 output (model 1 embeddings): {text}")

    logits = decode(model2, 50, temperature, 'cuda', prompt_embedding2)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 2 output (model 2 embeddings): {text}")

    logits = decode(model2, 50, temperature, 'cuda', transformed_prompt_embedding2)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 2 output (model 1 embeddings projected): {text}")


def experiment2():
    '''
    In this experiment,
    '''
    temperature = 0.01
    print(f" ------- \n * temperature: {temperature}")

    for _ in range(20):
        prompt_random_embedding1 = torch.randn([1, 8, model1.config.n_embd]).to('cuda')
        print(" . . . ")

        transformed_prompt_embedding2 = torch.matmul(prompt_random_embedding1, transformation_1_to_2_matrix)

        logits = decode(model1, 50, temperature, 'cuda', prompt_random_embedding1)
        text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
        print(f" model 1 output (model 1 with random prompt): {text}")

        logits = decode(model2, 50, temperature, 'cuda', transformed_prompt_embedding2)
        text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
        print(f" model 2 output (model 1 embeddings projected): {text}")


experiment1()
# experiment2()
