from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import wandb
import math
from utils import one_hot, decode_with_embedding, get_text_from_logits

wandb.init(project='embedding projection')

device = 'cuda'
model1_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model1_size)

model1 = GPT2LMHeadModel.from_pretrained(model1_size, output_hidden_states=True)
model1.to(device)
model1.eval()

input_ids = tokenizer.encode("To travel to Canada", return_tensors="pt").to(device)
input_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
# context_length = input_ids.size()[1]

prompt_embedding1 = torch.matmul(input_one_hot.type(torch.FloatTensor).to(device),
                                 model1.get_input_embeddings().weight)

l1 = torch.norm(prompt_embedding1, p=1)
l2 = torch.norm(prompt_embedding1, p=2)
print(f" l1 norm: {l1}")
print(f" l2 norm: {l2}")

temperature = 0.001
for order in range(10):
    variance = math.pow(10, -order)
    print(" - - - - - ")
    print(f" Variance: {variance}")
    noise = torch.randn(prompt_embedding1.size()).to(device) * variance
    logits = decode_with_embedding(model1, 50, temperature, device, prompt_embedding1 + noise)
    text, nll, _ = get_text_from_logits(logits[0, :, :], tokenizer)
    print(f" Model output, given prompt: {text}")
