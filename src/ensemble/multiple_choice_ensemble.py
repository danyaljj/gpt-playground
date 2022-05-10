import json
import argparse
from typing import Optional, Union, Tuple
import torch

torch.manual_seed(0)

from transformers import BertModel, BertTokenizer, PreTrainedModel, BertConfig
from transformers.modeling_outputs import MultipleChoiceModelOutput

# cache_dir = '/gscratch/xlab/msclar/'
# if not torch.cuda.is_available():
#     cache_dir = './'
# cache_dir = os.path.join(cache_dir, '.cache')

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


class EnsembledBertForMultipleChoice(PreTrainedModel):
    config_class = EnsembledBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert_models = torch.nn.ModuleList(
            [BertModel(config) for _ in range(self.config.num_models)]
        )  # initialize with empty models

        # MSCLAR: simplest is a linear layer, check bounds of this and then also include some non-linearity nn.ReLU()
        if self.config.non_linearity:
            hidden_size = len(self.bert_models) * self.config.hidden_size // 5
            self.cls = torch.nn.Sequential(
                torch.nn.Linear(len(self.bert_models) * self.config.hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1)
            )
        else:
            self.cls = torch.nn.Linear(len(self.bert_models) * self.config.hidden_size, 1)

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

        print('forward')
        for i in range(len(self.bert_models)):
            print(self.bert_models[i].encoder.layer[0].output.dense.bias[:10])
        print('')

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        input_ids_1 = input_ids.view(-1, input_ids.size(-1)).to(device) if input_ids is not None else None
        attention_mask_1 = attention_mask.view(-1, attention_mask.size(-1)).to(
            device) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)).to(
            device) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)).to(device) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)).to(device)
            if inputs_embeds is not None
            else None
        )

        outputs = [
            self.bert_models[i](
                input_ids_1,
                attention_mask=attention_mask_1,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ) for i in range(len(self.bert_models))
        ]

        # last layer of hidden state (top layer of BERT), and 0 since the [CLS] token goes first
        # original code uses pooler_output, but I think we don't want this because it adds an extra layer on top
        # https://github.com/huggingface/transformers/issues/7540
        last_hidden_states = torch.cat([out.last_hidden_state[:, 0, :] for out in outputs],
                                       dim=1)  # [batch_size, 2 * hidden_size]
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
            hidden_states=torch.cat([out.hidden_states for out in outputs], dim=-1) if all(
                [out.hidden_states for out in outputs]) else None,
            attentions=torch.cat([out.attentions for out in outputs], dim=-1) if all(
                [out.attentions for out in outputs]) else None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_models', type=int, default=2)
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--non_linearity', action='store_true')
    parser.add_argument('--model_names_list', type=str, default='')
    parser.add_argument('--freeze_bert_models', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()
    print(json.dumps(vars(args)))

    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bert_config = BertModel.from_pretrained("google/multiberts-seed_0").config

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
