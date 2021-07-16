from torch.distributions import Categorical
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
import wandb

wandb.init(project='embedding projection')


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


device = 'cuda'

model1_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model1_size)

model1 = GPT2LMHeadModel.from_pretrained(model1_size, output_hidden_states=True)
model1.to(device)
model1.eval()

model2_size = "gpt2-medium"
model2 = GPT2LMHeadModel.from_pretrained(model2_size, output_hidden_states=True)
model2.to(device)
model2.eval()

input_ids = tokenizer.encode("To travel to Canada", return_tensors="pt").to(device)
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
    prompt_embedding1 = torch.matmul(input_one_hot.type(torch.FloatTensor).to(device),
                                     model1.get_input_embeddings().weight)
    prompt_embedding2 = torch.matmul(input_one_hot.type(torch.FloatTensor).to(device),
                                     model2.get_input_embeddings().weight)

    print(" . . . ")
    transformed_prompt_embedding2 = torch.matmul(prompt_embedding1, transformation_1_to_2_matrix)

    # compare the embeddings
    diff = torch.abs(torch.mean(transformed_prompt_embedding2 - prompt_embedding2)).tolist()
    assert diff < 0.01, f'Diff: {diff} - the projection did not work! :-/ '

    temperature = 0.01  # to simulate the effect of arg-max
    logits = decode(model1, 50, temperature, device, prompt_embedding1)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 1 output (proper prompt): {text}")

    logits = decode(model2, 50, temperature, device, prompt_embedding2)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 2 output (proper prompt): {text}")

    logits = decode(model2, 50, temperature, device, transformed_prompt_embedding2)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 2 output (projected prompt): {text}")


def experiment2():
    '''
    In this experiment,
    '''
    temperature = 0.01
    print(f" ------- \n * temperature: {temperature}")

    for _ in range(20):
        prompt_random_embedding1 = torch.randn([1, 8, model1.config.n_embd]).to(device)
        print(" . . . ")

        transformed_prompt_embedding2 = torch.matmul(prompt_random_embedding1, transformation_1_to_2_matrix)

        logits = decode(model1, 50, temperature, device, prompt_random_embedding1)
        text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
        print(f" model 1 output (model 1 with random prompt): {text}")

        logits = decode(model2, 50, temperature, device, transformed_prompt_embedding2)
        text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
        print(f" model 2 output (model 1 embeddings projected): {text}")


def experiment3():
    '''
    In this experiment, we check if the embedding matrices are invertible
    '''
    E1 = model1.get_input_embeddings().weight
    E1_inv = torch.pinverse(E1)

    # try right inverse
    t1 = torch.matmul(E1, E1_inv)  # this is not diagonal

    # eye1 = torch.eye(t1.size()[0], t1.size()[1]).to('cuda') # runs out of memory here
    # diff = torch.mean(torch.abs(t1 - eye1))
    # print(diff)

    t2 = torch.matmul(E1_inv, E1)  # this is diagonal
    # eye2 = torch.eye(t2.size()[0], t2.size()[1]).to('cuda')
    # diff = torch.mean(torch.abs(t2 - eye2))

    # E1_t =  torch.transpose(E1, 0, 1)
    # Esquared1 = torch.pinverse(torch.matmul(E1_t, E1))
    # Esquared2 = torch.pinverse(torch.matmul(E1, E1_t))


class EmbeddingProjection(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(EmbeddingProjection, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)

    def transform(self, x):
        return self.linear(x)

    def forward(self, x, gold_projected):
        projected = self.linear(x)
        # TOOD: add a condition for loss computation
        # alternatively: nn.L1Loss
        # loss = torch.nn.MSELoss()(projected, gold_projected)
        loss = torch.nn.L1Loss()(projected, gold_projected)
        return projected, loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, inputSize, outputSize):
        model = EmbeddingProjection(inputSize, outputSize)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


def experiment4():
    '''
    Here we tune a linear transformation that maps the embeddings of a model to the embeddings of a different model
    '''
    embeddings1 = model1.get_input_embeddings().weight
    embeddings2 = model2.get_input_embeddings().weight

    # the parameters that we're optimizing
    w = EmbeddingProjection(model1.config.n_embd, model2.config.n_embd).to(device)

    lr = 0.001
    step_size = 1
    optim = torch.optim.Adam(w.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size)

    for iter in range(20):
        _, _loss = w.forward(embeddings1, embeddings2)
        _loss.backward(retain_graph=True)
        print(_loss)
        # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
        optim.step()
        scheduler.step()

        grad_norms = [p.grad.data.norm(2).tolist() for p in
                      list(filter(lambda p: p.grad is not None, w.parameters()))]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

        wandb.log({
            'iter': iter,
            "loss": _loss.detach().tolist(),
            "log_loss": torch.log(_loss).detach().tolist(),
            'avg_grad_norm': avg_grad_norm,
        })

        w.zero_grad()
    w.save('linear_transfer_v1.model')


def experiment5():
    '''
    Here we load the transformation of experiment 4 and apply it on a prompt
    '''
    prompt_embedding1 = torch.matmul(input_one_hot.type(torch.FloatTensor).to(device),
                                     model1.get_input_embeddings().weight)
    prompt_embedding2 = torch.matmul(input_one_hot.type(torch.FloatTensor).to(device),
                                     model2.get_input_embeddings().weight)


    # embeddings1 = model1.get_input_embeddings().weight
    # embeddings2 = model2.get_input_embeddings().weight

    temperature = 0.001

    logits = decode(model1, 50, temperature, device, prompt_embedding1)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 1 output (proper prompt): {text}")

    logits = decode(model2, 50, temperature, device, prompt_embedding2)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 2 output (proper prompt): {text}")

    w = EmbeddingProjection.load('linear_transfer_v1.model', model1.config.n_embd, model2.config.n_embd).to(device)
    projected_embedding2 = w.transform(prompt_embedding1)
    logits = decode(model2, 50, temperature, device, projected_embedding2)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" model 2 output (projected prompt): {text}")


# experiment1()
# experiment2()
# experiment3()
experiment4()
experiment5()
