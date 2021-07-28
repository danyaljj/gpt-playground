from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import wandb
import torch.nn.functional as F
from utils import one_hot, decode_with_embedding, get_text_from_logits, decode_with_one_hot
from os import listdir

wandb.init(project='discrete prompt from continuous')

device = 'cuda'

model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)

model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to(device)
model.eval()


def discrete_prompt_from_continuous(prompt_embedding, max_iter=100000, verbose = True):
    '''
    This experiment finds a discrete representation of a prompt.
    The objective consistens of the following terms:
        - min_{L} norm(e - L x E), for fixed e and E
        - min_{L} entropy(L) so that we have minimum entropy for L
        - while maximzing some measure of coherence for L (like NLL loss).
    '''

    # the parameters that we're optimizing
    batch_size = 1
    prefix_length = prompt_embedding.size()[1]  # this has to match the size of the embedding
    if verbose:
        print(f" * prompt length: {prefix_length}")
    optimized_word_logits = torch.nn.Parameter(
        torch.rand([batch_size, prefix_length, tokenizer.vocab_size], device='cuda')
    )

    lr = 100
    step_size = 500
    optimizer = torch.optim.Adam([optimized_word_logits], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.89)
    temperature = 0.01
    for iter in range(max_iter):
        norm = torch.nn.MSELoss()
        # norm = torch.nn.L1Loss()
        optimized_word_probs = torch.nn.Softmax(dim=2)(optimized_word_logits / temperature)
        # try straigh-through
        if True:
            optimized_word_probs_straight_through = (optimized_word_probs - optimized_word_logits).detach() + optimized_word_logits
        dist = norm(prompt_embedding, torch.matmul(optimized_word_probs_straight_through, model.get_input_embeddings().weight))
        entropy = torch.mean(
            # entropy for each position
            torch.sum(-torch.log(optimized_word_probs + 0.000001) * optimized_word_probs, dim=2)
        )
        _loss = dist
        _loss.backward(retain_graph=True)

        if iter % 500 == 0 or iter == max_iter - 1:
            if verbose:
                print(f" - - - - - - - \n iter = {iter}")
                print(f" loss: {_loss}")
                # print(f"temperature: {dynamic_temperature}")
                temperature = 0.001
                logits = decode_with_embedding(model, 50, temperature, device, prompt_embedding)
                text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
                print(f" model prediction (originl prompt): {text}")

            logits = decode_with_one_hot(model, 50, optimized_word_logits, temperature, device)
            text_prediction_using_optimized_logits, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
            if verbose:
                print(f" model prediction (using optimized logits): {text_prediction_using_optimized_logits}")

            text_optimized_logits, nll, _ = get_text_from_logits(optimized_word_logits[0, :, :], tokenizer)
            if verbose:
                print(f" the optimized logits: {text_optimized_logits}")

            if 'optimized_word_probs' in globals() and verbose:
                optimized_prompt_embedding = torch.matmul(optimized_word_probs_straight_through,
                                                          model.get_input_embeddings().weight)
                logits = decode_with_embedding(model, 50, temperature, device, optimized_prompt_embedding)
                text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
                print(f" model prediction (using predicted embeddings): {text}")

                text, nll, _ = get_text_from_logits(optimized_word_probs[0, :, :], tokenizer)
                print(f" the optimized probs: {text}")

            if iter == max_iter - 1:
                return _loss.detach().tolist(), text_optimized_logits, text_prediction_using_optimized_logits

        # torch.nn.utils.clip_grad_norm_([optimized_word_probs], 1.0)
        optimizer.step()
        scheduler.step()

        grad_norms = [p.grad.data.norm(2).tolist() for p in
                      list(filter(lambda p: p.grad is not None, model.parameters()))]
        avg_grad_norm = sum(grad_norms) / len(grad_norms)

        grad_sum = sum(
            [torch.sum(p.grad.data).tolist() for p in list(filter(lambda p: p.grad is not None, model.parameters()))])

        wandb.log({
            'iter': iter,
            "loss": _loss.detach().tolist(),
            "l1": dist.detach().tolist(),
            "entropy": entropy.detach().tolist(),
            "log_loss": torch.log(_loss).detach().tolist(),
            'avg_grad_norm': avg_grad_norm,
            'grad_sum': grad_sum
        })

        optimizer.zero_grad()

    # model.save('linear_transfer_v1.model')


def experiment1():
    input_ids = tokenizer.encode(
        "While the best food in Seattle is kebab, there are other form of garbage sentences that one can extract in order to",
        return_tensors="pt").to(device)
    input_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
    prompt_embedding = torch.matmul(input_one_hot.type(torch.FloatTensor).to(device),
                                    model.get_input_embeddings().weight)
    discrete_prompt_from_continuous(prompt_embedding)


def experiment2():
    dir = '/tmp/gpt-playground/optimized_prompts/'
    onlyfiles = [f for f in listdir(dir)]
    print(f"Available prompts: {onlyfiles}")
    prompt_embedding = torch.load(dir + 'optimized_prompt_jumped_to_bite.pt').to(device)
    discrete_prompt_from_continuous(prompt_embedding)


def experiment3():
    '''
    sensitivity to continuous prompts
    '''

    input_ids = tokenizer.encode(
        "While the best food in Seattle is kebab, there are other form of garbage sentences that one can extract in order to",
        return_tensors="pt").to(device)
    input_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
    prompt_embedding = torch.matmul(input_one_hot.type(torch.FloatTensor).to(device),
                                    model.get_input_embeddings().weight)

    for idx in range(10):
        noise_level = 0.1 * idx
        print(" - - - ")
        print(noise_level)
        prompt_embedding += noise_level * torch.randn(size=prompt_embedding.size()).to(device)
        loss, text_optimized_logits, text_pred_via_optimized_logits = discrete_prompt_from_continuous(prompt_embedding, max_iter=500, verbose=False)
        print(loss, text_optimized_logits, text_pred_via_optimized_logits)


# experiment1()
# experiment2()
experiment3()
