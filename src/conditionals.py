import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
import json
import redis

model_name = 'gpt2'
model = None
tokenizer = None
device = 'cpu'

# starting redis client: https://gist.github.com/tomysmile/1b8a321e7c58499ef9f9441b2faa0aa8
redis_client = redis.Redis(host='localhost', port=6379, db=0)


def load_model():
    global model
    global tokenizer
    print(" ==> Loading the models . . . ")
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def scores_to_tokens(scores):
    '''expect `$logits` to be a single column of logits '''
    assert len(scores[0]) == tokenizer.vocab_size
    top_score, top_idx = torch.topk(scores, k=1, dim=1)
    # print(torch.topk(logits, k=20, dim=1))
    text = tokenizer.batch_decode(top_idx.tolist())
    print((top_score, top_idx, text))
    return text


def lr_conditional_prob(context_ids, context_attention_mask=None):
    '''
    heads up: this function does not work for sequences of different size (if they're padded, right or left).
    there is some weird stuff about positional-ids/embeddings that I can't grasp.
    '''
    out = model(input_ids=context_ids)
    logits = out.logits[:, -1, :]
    probs = F.softmax(logits, dim=1)
    return probs


def prior_prob():
    if False:
        # uniform probability
        return torch.ones(size=[tokenizer.vocab_size]) / tokenizer.vocab_size
    else:
        for word_id in tqdm(range(tokenizer.vocab_size)):
            with open('/Users/danielk/ideaProjects/Refinement-Generation/priors_1000.jsonl', 'a+') as outfile:
                prior = prior_prob_for_word(word_id)
                outfile.write(
                    json.dumps(
                        {'id': word_id, 'prior': prior, 'token': tokenizer.convert_ids_to_tokens(word_id)}) + "\n"
                )


def prior_prob_for_word(word_id):
    probs1 = redis_client.get(word_id).decode().split(" ")
    assert len(probs1) == tokenizer.vocab_size, f" * A: the vocab len: {len(probs1)} vs {tokenizer.vocab_size}"
    cache = 0.0
    # look at the subset
    skip = 50
    for prefix_id in range(0, tokenizer.vocab_size, skip):
        probs2 = redis_client.get(prefix_id).decode().split(" ")
        assert len(probs2) == tokenizer.vocab_size, f" * B: the vocab len: {len(probs2)} vs {tokenizer.vocab_size}"
        x = float(probs1[prefix_id])
        y = float(probs2[word_id])
        if prefix_id == word_id:
            assert x == y, 'x and y are not the same, for the same ids. Something is off! :-/ '
        cache += x / y
    prior = 1 / (skip * cache)
    print(prior)
    return prior


def cache_bigrams():
    batch_size = 64
    for prefix_ids in tqdm(batch_iterable(range(tokenizer.vocab_size), batch_size)):
        base_word_ids = torch.LongTensor([[x] for x in prefix_ids]).to(device)
        probs = lr_conditional_prob(base_word_ids)
        for i, idx in enumerate(prefix_ids):
            lst = probs[i, :].detach().tolist()
            lst = " ".join(["{:.1e}".format(x) for x in lst])
            redis_client.set(idx, lst)


def rl_conditional_prob_fix_left(context_ids, right_ids):
    '''
    computes dist over prefixes that that lead to a certain word on the right
    '''
    assert right_ids.size()[1] == 1, f"must be a single word: {right_ids.size()}"

    batch_size = 512
    batch_context_ids = context_ids.repeat(batch_size, 1)
    all_probs = []

    target_vocab = batch_iterable(range(tokenizer.vocab_size), batch_size)
    for prefix_ids in tqdm(target_vocab):
        if len(prefix_ids) < batch_size:
            batch_context_ids = context_ids.repeat(len(prefix_ids), 1)

        prefix_ids_tensor = torch.LongTensor([[x] for x in prefix_ids]).to(device)
        prepended_batch_context_ids = torch.cat((prefix_ids_tensor, batch_context_ids), dim=1)

        # use the lr sub-routine to try all the possible optons
        probs = lr_conditional_prob(context_ids=prepended_batch_context_ids)
        right_context_probs = probs[:, right_ids[0][0]].detach()  # TODO: I think we should look at the first index
        all_probs += right_context_probs.tolist()

    return torch.Tensor([all_probs])


def rl_conditional_prob(context):
    '''
    right to left generation using left to right generation
    :param context:
    :return:
    '''
    input_ids = tokenizer.encode(context, return_tensors="pt")
    current_prob = prior_prob()
    for idx in range(len(input_ids)):
        lr_conditional_prob


load_model()

# test lr with a single example
encoded_input = tokenizer("she is a", return_tensors="pt")
assert scores_to_tokens(lr_conditional_prob(encoded_input['input_ids'].to(device))) == [" very"]

if False:
    encoded_input = tokenizer("I am a", return_tensors="pt")
    assert scores_to_tokens(lr_conditional_prob(encoded_input['input_ids'].to(device))) == [" very"]

    encoded_input = tokenizer("A green", return_tensors="pt", device=device)
    assert scores_to_tokens(lr_conditional_prob(encoded_input['input_ids'].to(device))) == [' light']

    # test lr with batched examples
    encoded_input = tokenizer(["I am a", 'She is a', 'He is a', 'She will be'], return_tensors="pt")
    assert scores_to_tokens(lr_conditional_prob(
        encoded_input['input_ids'].to(device)
    )) == [' very', ' very', ' very', ' the']

# note the space before context and right token
context_ids = tokenizer.encode("", return_tensors="pt").to(device).type(torch.LongTensor)
right_ids = tokenizer.encode(" bark", return_tensors="pt").to(device)
probs = rl_conditional_prob_fix_left(context_ids=context_ids, right_ids=right_ids)
print(scores_to_tokens(probs))

# cache_bigrams()

# The_id = tokenizer.encode(" The", return_tensors="pt").detach().tolist()[0][0]
# the_id = tokenizer.encode(" the", return_tensors="pt").detach().tolist()[0][0]
# dot_id = tokenizer.encode(".", return_tensors="pt").detach().tolist()[0][0]
# apple_id = tokenizer.encode(" apple", return_tensors="pt").detach().tolist()[0][0]
# d_id = tokenizer.encode(" d", return_tensors="pt").detach().tolist()[0][0]
#
# prior_prob_for_word(The_id)
# prior_prob_for_word(the_id)
# prior_prob_for_word(dot_id)
# prior_prob_for_word(apple_id)
# prior_prob_for_word(d_id)

# prior_prob()
