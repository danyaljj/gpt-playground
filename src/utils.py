import torch
import torch.nn.functional as F
from torch import nn

# used to prevent numerical issues
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

eps = 1e-10


def decode_with_embedding(model, length, temperature, device, prompt_embedding):
    '''
    GPT2 decoding via dense representations (no arg-max)
    '''
    past = None
    inputs_embeds = None
    logits_so_far = None
    for i in range(length):
        if past is None:
            inputs_embeds = prompt_embedding
        model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds)
        logits = model_outputs.logits
        past = model_outputs.past_key_values
        logits = logits[:, -1, :] / temperature
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device=device)
    return logits_so_far


def decode_with_one_hot(model, length, input_one_hot1, temperature, device):
    '''
    GPT2 decoding via dense representations (no arg-max)
    '''
    past = None
    inputs_embeds1 = None
    logits_so_far = None
    for i in range(length):
        if past is None:
            # inputs_embeds = model.transformer.wte(input_ids)
            inputs_embeds1 = embed_inputs(model.get_input_embeddings(),
                                          input_one_hot1.type(torch.FloatTensor) / temperature, device='cuda',
                                          print_entropy=True)
        model_outputs = model(past_key_values=past, inputs_embeds=inputs_embeds1)
        logits = model_outputs.logits
        past = model_outputs.past_key_values
        logits = logits[:, -1, :] / temperature
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        inputs_embeds1 = embed_inputs(model.get_input_embeddings(), logits, device=device)
    return logits_so_far


def decode_with_argmax(model, length, input_ids1, device):
    '''
    GPT2 decoding via dense representations (no arg-max)
    '''
    logits_so_far = None
    for i in range(length):
        # if past is None:
        # inputs_embeds = model.transformer.wte(input_ids1)
        # inputs_embeds = embed_inputs(model.get_input_embeddings(), input_one_hot.type(torchc)/ temperature, device='cuda', print_entropy=True)
        model_outputs = model(input_ids=input_ids1)
        # else:
        #     model_outputs = model(past_key_values=past, input_ids=input_ids1)
        logits = model_outputs.logits
        # past = model_outputs.past_key_values
        logits = logits[:, -1, :]
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        # inputs_embeds = embed_inputs(model.get_input_embeddings(), logits, device=device)
        next_token = torch.argmax(logits)
        input_ids1 = torch.cat([input_ids1, next_token.unsqueeze(0).unsqueeze(0)], 1)
    return logits_so_far,


# given a sentence return an embedded version of it
def embed_sentence(model, input_ids, device):
    logits = decode_with_argmax(model, input_ids.shape[1], input_ids, device=device)
    # use the logits of the last word
    logits = logits[0][0][-1]
    return embed_inputs(model.get_input_embeddings(), logits, device=device)


def embed_inputs(embedding, logits, device, print_entropy=False):
    '''
    embeds inputs in a dense representation, before passing them to the model
    '''
    # typically we embed a one-hot vector. But here since we work we work with dense representations,
    # we have softmax here to make sure that all the values of the input logits sum to one (similar to a 1-hot vector).
    probs = F.softmax(logits, dim=-1)
    # probs = logits
    if print_entropy:
        _entropy = - probs * torch.log(probs)
        _entropy = torch.sum(_entropy)
        print(_entropy)

    probs = probs.to(device)
    return torch.matmul(probs, embedding.weight)


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
    text = tokenizer.decode(output_so_far.tolist())
    text = text.replace('\n', ' ')
    return text, nll, output_so_far


def one_hot(tensor, dimension):
    while len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    onehot = torch.LongTensor(tensor.shape[0], tensor.shape[1], dimension).to(tensor.device)
    onehot.zero_().scatter_(2, tensor.unsqueeze(-1).to(torch.int64), 1)
    return onehot


def plot_histogram(x):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.hist(x)
    plt.show()


def svd_model_embeddings(model):
    '''
    uses SVD to decompose
    '''
    E = model.get_input_embeddings().weight
    # since the embedding matrix is not degenerate, `full_matrices` should not have any effect in the size of the matrices
    u, s, vh = torch.linalg.svd(E, full_matrices=False)

    # verify that this is a good decomposition
    r = torch.matmul(u, s * vh)
    diff = torch.mean(torch.abs(r - E))
    assert diff < 0.07, f"the diff is larger than expected {diff}"

    return u, s, vh


def project_ids(input_ids, model, tokenizer, device):
    desired_beginning_ids = tokenizer.encode(input_ids, return_tensors="pt").to(device)
    desired_beginning_one_hot = one_hot(desired_beginning_ids, dimension=tokenizer.vocab_size)
    embeddings = torch.matmul(desired_beginning_one_hot.type(torch.FloatTensor).to(device),
                              model.get_input_embeddings().weight.to(device))
    return embeddings


def project_embeddings(embedding, model, temp, with_streight_through=False):
    logits = torch.matmul(embedding, torch.transpose(model.get_input_embeddings().weight, 0, 1))
    # if with_streight_through:
    #     logits = (logits.detach() / temp - logits).detach() + logits
    # else:
    #     logits = logits / temp
    probs = F.softmax(logits, dim=-1)
    # if with_streight_through:
    #     probs = (probs.detach() - logits).detach() + logits

    return probs


def new_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
        ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.transformer.first_device)
        hidden_states = hidden_states.to(self.lm_head.weight.device)

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    output = CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
        cross_attentions=transformer_outputs.cross_attentions,
    )
    output.hidden_states = hidden_states  # new addition
    return output
