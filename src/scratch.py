from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

# model_size = "gpt2-xl"
model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to('cuda')
model.eval()
input_ids = tokenizer.encode("A new study is coming to the University of Michigan this fall.", return_tensors="pt").to('cuda')
# input_ids = tokenizer.encode("Jack was enjoying his game. ", return_tensors="pt")
# outputs = model.generate(input_ids)
# print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
outputs = model(input_ids)
# print(outputs)
output = model(input_ids=input_ids, labels=input_ids)
logits = output.logits
logits = logits.squeeze()
logits.squeeze()[input_ids.squeeze(), :]
loss = output.loss
loss = loss.detach().clone().data.cpu().numpy()
ppl = np.exp(loss)


