from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from utils import one_hot, embed_inputs, decode_with_one_hot, get_text_from_logits, decode_with_argmax


def query_via_embeddings():
    input_ids = tokenizer("In my early life", return_tensors='pt')['input_ids'].to('cuda')
    inputs_embeds = model.transformer.wte(input_ids).squeeze()

    out1 = model(inputs_embeds=inputs_embeds)
    out2 = model(input_ids=input_ids)

    input_ids_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)
    input_embeddings2 = embed_inputs(model.get_input_embeddings(), input_ids_one_hot.type(torch.FloatTensor),
                                     device='cuda')
    out3 = model(inputs_embeds=input_embeddings2)
    print(out1)
    print(out2)
    print(out3)


model_size = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
model = GPT2LMHeadModel.from_pretrained(model_size, output_hidden_states=True)
model.to('cuda')
model.eval()
input_ids = tokenizer.encode("In order to make an omelette", return_tensors="pt").to('cuda')
input_one_hot = one_hot(input_ids, dimension=tokenizer.vocab_size)


def experiment1():
    '''
    in this experiment, we assess the connection between the input peakiness and the quality of the output generations
    lower temperature results in peakier prompts
    '''
    for temperature in [0.001, 0.01, 0.1, 0.2, 0.3, 1, 5]:
        print(f" ------- \n * temperature: {temperature}")
        logits_so_far = decode_with_one_hot(model, 100, input_one_hot, temperature, 'cuda')
        text, nll, _ = get_text_from_logits(logits_so_far[0, :, :], tokenizer)
        print(text)


def experiment2():
    '''
    in this experiment, we try the conventional greedy decopding of GPT (argmax at each step).
    '''
    logits_so_far = decode_with_argmax(model, 100, input_ids, 'cuda')
    text, nll, _ = get_text_from_logits(logits_so_far[0, :, :], tokenizer)
    print(text)


experiment2()
