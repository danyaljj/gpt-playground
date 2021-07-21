import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wandb

wandb.init(project='reverse decoding')

model_name = 'gpt2'
model = None
tokenizer = None
device = 'cuda'


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
    text = tokenizer.decode_with_one_hot(output_so_far.tolist())
    text = text.replace('\n', ' ')
    return text, nll, output_so_far


def load_model():
    global model
    global tokenizer
    print(" ==> Loading the models . . . ")
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.eos_token
    # model.config.pad_token_id = model.config.eos_token_id


def conditional_prob_with_ids(context_ids):
    '''
    heads up: this function does not work for sequences of different size (if they're padded, right or left).
    there is some weird stuff about positional-ids/embeddings that I can't grasp.
    '''
    out = model(input_ids=context_ids)
    logits = out.logits
    return logits


def conditional_prob_with_embdddings(inputs_embeds):
    '''
    heads up: this function does not work for sequences of different size (if they're padded, right or left).
    there is some weird stuff about positional-ids/embeddings that I can't grasp.
    '''
    out = model(inputs_embeds=inputs_embeds)
    logits = out.logits
    # probs = F.softmax(logits, dim=0)
    return logits


load_model()

batch_size = 5
prefix_length = 1
# it actually does not work for sentence_len == 1; the issue reported
for sentence_len in range(2, 10):
    sentence_str = " ".join(["good"] * sentence_len)
    input_ids = tokenizer([sentence_str] * batch_size, return_tensors='pt')['input_ids'].to(device)
    inputs_embeds = model.transformer.wte(input_ids).squeeze()

    out1 = model(inputs_embeds=inputs_embeds)
    out2 = model(input_ids=input_ids)

    # make sure that querying a model via ids and embeddings leacd to the same result
    diff = torch.abs(torch.mean(out1.logits - out2.logits))
    print(f" - sentence length: {sentence_len} <-> {diff}")
    assert diff < 0.001, f"The difference is bigger than expected: {diff}"

input_ids = tokenizer(['is mine.'] * batch_size, return_tensors='pt')['input_ids'].to(device)
inputs_embeds = model.transformer.wte(input_ids).squeeze()

optimized_embeddings = torch.nn.Parameter(torch.rand([batch_size, prefix_length, model.config.n_embd], device=device))

lr = 0.001
step_size = 1
optim = torch.optim.Adam([optimized_embeddings], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=step_size)
temperature = 0.01

for iter in range(1000):
    past = None
    inputs_embeds = None

    logits = conditional_prob_with_embdddings(optimized_embeddings)

    # TODO: if the gold prediction is not in top-k (e.g., k == 1), punish bigly
    # compute loss with respect to the ending
    right_context_probability = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    _loss = right_context_probability
    _loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
    optim.step()
    scheduler.step()

    print(" - - - - ")
    for batch_id in range(batch_size):
        predicted, nll, _ = get_text_from_logits(logits[batch_id, :, :], tokenizer)
        print(f" * prefix: <--> {predicted}")

    grad_norms = [p.grad.data.norm(2).tolist() for p in list(filter(lambda p: p.grad is not None, model.parameters()))]
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

    wandb.log({
        "total_loss": _loss.detach().tolist(),
        "_dist": right_context_probability.detach().tolist(),
        'avg_grad_norm': avg_grad_norm,
    })

    model.zero_grad()
