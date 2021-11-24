from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import get_text_from_logits
import torch
import utils

GPT2LMHeadModel.forward = utils.new_forward
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = 'cuda'
batch_size = 16
torch.manual_seed(43)


def avg_output_embedding_distance(model_size, initial_prompts):

    model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
    model.to(device)
    model.eval()
    print(f" ---------- \n Model: {model_size}")
    for initial_prompt in initial_prompts:
        print(f"Prompt: {initial_prompt}")
        initial_prompt_ids = tokenizer.encode(initial_prompt, return_tensors="pt").to(device)
        prefix_length = initial_prompt_ids.size()[1]  # the length of the provided phrase

        if True:
            inputs_embeds = model.transformer.wte(initial_prompt_ids)
            inputs_embeds_batched = inputs_embeds.repeat(batch_size, 1, 1)
            inputs_embeds_batched_randomized = inputs_embeds_batched + 0.001 * torch.rand(
                [batch_size, prefix_length, model.config.n_embd], device=device)
            probs_of_embeddings = utils.project_embeddings(inputs_embeds_batched_randomized, model, temp=0.001)
            # for batch_id in range(min(5, batch_size)):
            #     optimized_prefix, _, _ = get_text_from_logits(probs_of_embeddings[batch_id, :, :], tokenizer)
            #     print(f" * Projection of the perturbed sentences: {optimized_prefix}")
            inputs_embeds = torch.nn.Parameter(inputs_embeds_batched_randomized).to(device)
        else:
            # given a random set of continuous prompts (point cloud within unit sphere), compute the resulting embeddings and compute their diameter
            inputs_embeds = torch.nn.Parameter(torch.rand([batch_size, prefix_length, model.config.n_embd], device=device))
        model_outputs = model(past_key_values=None, inputs_embeds=inputs_embeds)
        avg_distance = []
        for position_idx in range(prefix_length):
            avg_distance.append(
                torch.mean(torch.cdist(
                    model_outputs.hidden_states[:, position_idx, :],
                    model_outputs.hidden_states[:, position_idx, :],
                    p=1)).cpu().detach().numpy().tolist()
            )
        # m1 = torch.mean(torch.norm(model.lm_head.weight, dim=1))
        # m2 = torch.norm(model.lm_head.weight)

        print(f"{avg_distance}")


with open("sentences.txt", 'r') as f:
    initial_prompts = [line.replace("\n", '') for line in f.readlines()]
    avg_output_embedding_distance(model_size="distilgpt2", initial_prompts=initial_prompts)
    avg_output_embedding_distance(model_size="gpt2", initial_prompts=initial_prompts)
    avg_output_embedding_distance(model_size="gpt2-medium", initial_prompts=initial_prompts)
    avg_output_embedding_distance(model_size="gpt2-large", initial_prompts=initial_prompts)
    avg_output_embedding_distance(model_size="gpt2-xl", initial_prompts=initial_prompts)
