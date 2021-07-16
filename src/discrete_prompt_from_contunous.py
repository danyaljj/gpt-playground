from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import wandb
import torch.nn.functional as F
from utils import one_hot, decode_with_embedding, get_text_from_logits, decode_with_one_hot

wandb.init(project='discrete prompt from continuous')

device = 'cuda'

model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)

model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to(device)
model.eval()

input_ids = tokenizer.encode("To travel to Canada", return_tensors="pt").to(device)
input_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
context_length = input_ids.size()[1]

def experiment1():
    '''
    This experiment finds a discrete representation of a prompt.
    The objective consistens of the following terms:
        - min_{L} norm(e - L x E), for fixed e and E
        - min_{L} entropy(L) so that we have minimum entropy for L
        - while maximzing some measure of coherence for L (like NLL loss).
    '''
    prompt_embedding = torch.matmul(input_one_hot.type(torch.FloatTensor).to(device), model.get_input_embeddings().weight)


    # the parameters that we're optimizing
    batch_size = 1
    prefix_length = 4 # let's say, this is known
    optimized_word_logits = torch.nn.Parameter(
        torch.rand([batch_size, prefix_length, tokenizer.vocab_size], device='cuda')
    )

    lr = 1.0
    step_size = 200
    optimizer = torch.optim.Adam([optimized_word_logits], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.89)


    for iter in range(10000):
        norm = torch.nn.MSELoss()
        # norm = torch.nn.L1Loss()
        l1 = norm(prompt_embedding, torch.matmul(optimized_word_logits, model.get_input_embeddings().weight))
        word_probs = F.softmax(optimized_word_logits, dim=2)
        entropy = torch.mean(
            # entropy for each position
            torch.sum(-torch.log(word_probs + 0.000001) * word_probs, dim=2)
        )
        _loss = l1 + entropy

        _loss.backward(retain_graph=True)

        if iter % 100 == 0:
            print(_loss)

        if iter % 500 == 0:
            print(f" - - - - - - - \n iter = {iter}")
            temperature = 0.001
            logits = decode_with_embedding(model, 50, temperature, device, prompt_embedding)
            text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
            print(f" model predictiow (originl prompt): {text}")

            logits = decode_with_one_hot(model, 50, optimized_word_logits, temperature, device)
            text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
            print(f" model predictiow (using predicted logits): {text}")

            optimized_prompt_embedding = torch.matmul(optimized_word_logits, model.get_input_embeddings().weight)
            logits = decode_with_embedding(model, 50, temperature, device, optimized_prompt_embedding)
            text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
            print(f" model predictiow (using predicted embeddings): {text}")

            text, nll, _ = get_text_from_logits(optimized_word_logits[0, :, :], tokenizer)
            print(f" the predicted logits: {text}")



        # torch.nn.utils.clip_grad_norm_([optimized_logits], 1.0)
        optimizer.step()
        scheduler.step()

        grad_norms = [p.grad.data.norm(2).tolist() for p in
                      list(filter(lambda p: p.grad is not None, model.parameters()))]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if len(grad_norms) > 0 else 0.0

        wandb.log({
            'iter': iter,
            "loss": _loss.detach().tolist(),
            "l1": l1.detach().tolist(),
            # "l2": l2.detach().tolist(),
            "entropy": entropy.detach().tolist(),
            "log_loss": torch.log(_loss).detach().tolist(),
            'avg_grad_norm': avg_grad_norm,
        })

        # model.zero_grad()
        optimizer.zero_grad() # TODO: need to add this function for the rest of the examples that do not involve modules.

    # model.save('linear_transfer_v1.model')

experiment1()