import json
import argparse
from typing import Optional, Union, Tuple
import torch
import os
import torch.multiprocessing as mp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


torch.manual_seed(0)

from transformers import BertModel, BertTokenizer, PreTrainedModel, BertConfig
from transformers.modeling_outputs import MultipleChoiceModelOutput

#cache_dir = '/gscratch/xlab/msclar/'
#if not torch.cuda.is_available():
#     cache_dir = './'
#cache_dir = os.path.join(cache_dir, '.cache')


def print_gpu_utilization():
    for i in range(torch.cuda.device_count()):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU_{i} memory occupied: {info.used//1024**2} MB.")


class EnsembledBertConfig(BertConfig):
    def __init__(
            self,
            num_models=2,
            non_linearity=False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_models = num_models
        self.non_linearity = non_linearity


def function_without_self(var):
    i, device, bert_model, input_ids_copy, attention_mask_copy, token_type_ids_copy, return_dict = var
    output = bert_model(
        input_ids_copy,
        attention_mask=attention_mask_copy,
        token_type_ids=token_type_ids_copy,
        # position_ids=position_ids,
        # head_mask=head_mask,
        # inputs_embeds=inputs_embeds,
        # output_attentions=output_attentions,
        # output_hidden_states=output_hidden_states,
        # return_dict=return_dict,
        return_dict=return_dict
    )

    return output.last_hidden_state[:, 0, :].detach()


# MSCLAR trying to copy this: https://github.com/huggingface/transformers/blob/58d047a596a97fbb815acb3e657102bf1960b06a/src/transformers/models/t5/modeling_t5.py#L1263-L1266
class EnsembledBertForMultipleChoice(PreTrainedModel):
    config_class = EnsembledBertConfig
    is_parallelizable = True

    def __init__(self, config):
        super().__init__(config)

        visible_devices = [f'cuda:{idx}' for idx in range(torch.cuda.device_count())]
        self.devices = []
        for i in range(self.config.num_models):
            self.devices.append(visible_devices[i % len(visible_devices)])

        print(f"Using devices: {self.devices}")
        self.bert_models = torch.nn.ModuleList(
            [BertModel(config).to(self.devices[i]) for i in range(self.config.num_models)]
        )  # initialize with empty models
        print('require_grad', all(param.requires_grad for model in self.bert_models for param in model.parameters()))
        #for model in self.bert_models:
        #    for param in model.parameters():
        #        param.requires_grad = False

        self.bert_models_to_device()

        # MSCLAR: simplest is a linear layer, check bounds of this and then also include some non-linearity nn.ReLU()
        assert type(self.config.non_linearity) == bool, "non_linearity must be boolean but it is {}".format(type(self.config.non_linearity))
        if self.config.non_linearity:
            hidden_size = self.config.num_models * self.config.hidden_size // 5
            self.cls = torch.nn.Sequential(
                torch.nn.Linear(self.config.num_models * self.config.hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1)
            )
        else:
            self.cls = torch.nn.Linear(self.config.num_models * self.config.hidden_size, 1)
        self.cls = self.cls.to('cuda:0')

        self.model_parallel = True

    def bert_models_to_device(self):
        for i, device in enumerate(self.devices):
            self.bert_models[i] = self.bert_models[i].to(device)

    def initialize_with_existing_berts(self, model_names_list, num_models):
        print(" >>>>>>>> initialize >>>>>>>>> ")
        if model_names_list:
            model_names = [model_name for model_name in model_names_list.split(',')]
        else:
            model_names = [f"google/multiberts-seed_{i}" for i in range(num_models)]

        assert self.config.num_models == len(model_names), "Number of BERT models must match"

        for idx, model_name in enumerate(model_names):
            model1 = BertModel.from_pretrained(model_name)
            self.bert_models[idx].load_state_dict(model1.state_dict())

            # TODO: make this a parameter for our ablation study
            for param in self.bert_models[idx].parameters():
               param.requires_grad = False
            self.bert_models[idx] = self.bert_models[idx].to(self.devices[idx])
        print('require_grad', all(param.requires_grad for model in self.bert_models for param in model.parameters()))

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        # torch.cuda.empty_cache()
        # print_gpu_utilization()

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids_1 = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask_1 = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        )

        input_ids_copy = [input_ids_1.detach().to(self.devices[i], non_blocking=True) for i in range(len(self.bert_models))]
        attention_mask_copy = [attention_mask_1.detach().to(self.devices[i], non_blocking=True) for i in range(len(self.bert_models))]
        token_type_ids_copy = [token_type_ids.detach().to(self.devices[i], non_blocking=True) for i in range(len(self.bert_models))]

        # https://www.machinelearningplus.com/python/parallel-processing-python/
        pool = mp.Pool(mp.cpu_count())
        outputs = pool.map(function_without_self,
                 [(i, self.devices[i], self.bert_models[i], input_ids_copy[i],
                   attention_mask_copy[i],
                   token_type_ids_copy[i],
                   return_dict) for i in range(len(self.bert_models))])
        pool.close()

        """
        outputs = [
            self.bert_models[i](
                input_ids_copy[i],
                attention_mask=attention_mask_copy[i],
                token_type_ids=token_type_ids_copy[i],
                #position_ids=position_ids,
                #head_mask=head_mask,
                #inputs_embeds=inputs_embeds,
                #output_attentions=output_attentions,
                #output_hidden_states=output_hidden_states,
                #return_dict=return_dict,
                return_dict=return_dict
            ) for i in range(self.config.num_models)
        ]
        """

        # last layer of hidden state (top layer of BERT), and 0 since the [CLS] token goes first
        # original code uses pooler_output, but I think we don't want this because it adds an extra layer on top
        # https://github.com/huggingface/transformers/issues/7540
        # MSCLAR: .last_hidden_state.detach() is now called inside each process
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        last_hidden_states = torch.cat([out.to(device) for out in outputs], dim=1)  # [batch_size, 2 * hidden_size]
        logits = self.cls(last_hidden_states)  # [batch_size, 3]
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            labels = labels.to(device)  # [batch_size]
            loss_fct = torch.nn.CrossEntropyLoss()  # https://pytorch.org/docs/stable/nn.html#crossentropyloss
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (reshaped_logits,) + tuple([out[2:] for out in outputs])
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=None,  # MSCLAR: returning hidden_states is now broken (to fix, add it to .detach() in fun())
            attentions=None,  # MSCLAR: returning attentions is now broken (to fix, add it to .detach() in fun())
            #hidden_states=torch.cat([out.hidden_states for out in outputs], dim=-1) if all(
            #    [out.hidden_states for out in outputs]) else None,  # MSCLAR: now broken, but was unused anyways
            #attentions=torch.cat([out.attentions for out in outputs], dim=-1) if all(
            #    [out.attentions for out in outputs]) else None,  # MSCLAR: now broken, but was unused anyways
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_models', type=int, default=2)
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--non_linearity', action='store_true')
    parser.add_argument('--model_names_list', type=str, default='')
    parser.add_argument('--load_model', action='store_true')

    # torch.multiprocessing.set_start_method('spawn')

    # TODO: check that num_models / num_models_per_device <= num_gpus

    args = parser.parse_args()
    print(json.dumps(vars(args)))

    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bert_config = BertModel.from_pretrained("google/multiberts-seed_0").config

    print('torch.cuda.device_count()', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print('torch.cuda.get_device_name(i)', torch.cuda.get_device_name(i))

    path = f'ensembed_bert_{args.num_models}'
    if not args.load_model:
        # initialize an empty BERT ensemble
        non_linearity = True if args.non_linearity else False
        config = EnsembledBertConfig(num_models=args.num_models, non_linearity=non_linearity, **bert_config.to_dict())
        model = EnsembledBertForMultipleChoice(config)

        # assign the pre-trained weights
        model.initialize_with_existing_berts(args.model_names_list, args.num_models)

        model.save_pretrained(path)

        print(" ====> after saving ....")
        for i in range(len(model.bert_models)):
            print(model.bert_models[i].encoder.layer[2].attention.self.query.weight[:3])
        print('')

    else:
        model = EnsembledBertForMultipleChoice.from_pretrained(path)

        print("  ====>  after loading ....")
        for i in range(len(model.bert_models)):
            print(model.bert_models[i].encoder.layer[2].attention.self.query.weight[:3])
        print('')

    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    choice0 = "It is eaten with a fork and a knife."
    choice1 = "It is eaten while held in the hand."
    choice2 = "It is eaten while held in the elbow."
    labels = torch.tensor(0).unsqueeze(0)

    with torch.no_grad():
        encoding = tokenizer([prompt, prompt, prompt], [choice0, choice1, choice2], return_tensors="pt", padding=True)
        print(encoding.keys())
        outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)

        # the linear classifier still needs to be trained
        loss = outputs.loss
        logits = outputs.logits
        print(loss, logits)
