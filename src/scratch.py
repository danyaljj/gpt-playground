from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import utils
import torch
import torch.nn.functional as F

# model_size = "gpt2-xl"
model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
device = 'cuda'
model.to(device)
model.eval()
input_ids = tokenizer.encode('The big brown dog looked angry at the policeman because he was walking behind', return_tensors="pt").to('cuda')
# input_ids = tokenizer.encode("Jack was enjoying his game. ", return_tensors="pt")

outputs = model.generate(input_ids)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if False:
    outputs = model(input_ids)
    # print(outputs)
    output = model(input_ids=input_ids, labels=input_ids)
    logits = output.logits
    logits = logits.squeeze()
    logits.squeeze()[input_ids.squeeze(), :]
    loss = output.loss
    loss = loss.detach().clone().data.cpu().numpy()
    ppl = np.exp(loss)

desired_beginning = "Karen was assigned a roommate her first year"
desired_beginning_ids = tokenizer.encode(desired_beginning, return_tensors="pt").to(device)
desired_beginning_one_hot = utils.one_hot(desired_beginning_ids, dimension=tokenizer.vocab_size)
embeddings = torch.matmul(desired_beginning_one_hot.type(torch.FloatTensor).to(device), model.get_input_embeddings().weight.to(device))

logits = torch.matmul(embeddings, torch.transpose(model.get_input_embeddings().weight, 0, 1))
temp = 0.01
probs = F.softmax(logits / temp, dim=-1)

text, nll, _ = utils.get_text_from_logits(logits[0, :, :], tokenizer)
print(text)

text, nll, _ = utils.get_text_from_logits(probs[0, :, :], tokenizer)
print(text)
