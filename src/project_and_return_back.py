from transformers import GPT2Tokenizer, GPT2LMHeadModel
import utils

# model_size = "gpt2-xl"
model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
device = 'cuda'
model.to(device)
model.eval()

desired_beginning = "Karen was assigned a roommate her first year"

# project to embedding space
embeddings = utils.project_ids(desired_beginning, model, tokenizer, device)

# project it back to word space
probs = utils.project_embeddings(embeddings, model, temp=0.001)

# generate
text, nll, _ = utils.get_text_from_logits(probs[0, :, :], tokenizer)
print(text)
