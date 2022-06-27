import json
import argparse
from typing import Optional, Union, Tuple
import torch

torch.manual_seed(0)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# cache_dir = '/gscratch/xlab/msclar/'
# if not torch.cuda.is_available():
#     cache_dir = './'
# cache_dir = os.path.join(cache_dir, '.cache')

ALL_MODELS_SMALL = [
    "stanford-crfm/expanse-gpt2-small-x777",
    "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/alias-gpt2-small-x21",
    "stanford-crfm/caprica-gpt2-small-x81",
]

mapping_revision_from_model = {
    'stanford-crfm/alias-gpt2-small-x21': '41b0b17afa98a105d768d7a7e29a7c994cfe48dc',
    'stanford-crfm/battlestar-gpt2-small-x49': '3518225c1dac26db77462aa05ecb97a2cdd7a340',
    'stanford-crfm/caprica-gpt2-small-x81': 'e726b453a62eead96777d1c00e127ff3ef11e754',
    'stanford-crfm/darkmatter-gpt2-small-x343': '26347553202c0936b8147fa450d89cb0e6d8d661',
    'stanford-crfm/expanse-gpt2-small-x777': '74b5cd419d568e6f42ae470363227deec7328712',

    'stanford-crfm/arwen-gpt2-medium-x21': '68e01476dfb0bbbf3850a553a37c22c132aa4b71',
    'stanford-crfm/beren-gpt2-medium-x49': 'ebb67d0d5dcb4829fb358ee9db02c2485c82984b',
    'stanford-crfm/celebrimbor-gpt2-medium-x81': 'd76ea84327a8e25533f9683e480a8fb6b9f167be',
    'stanford-crfm/durin-gpt2-medium-x343': '3c72a1446d7a51dd686d91d5aa5b6db6f140e4f6',
    'stanford-crfm/eowyn-gpt2-medium-x777': '6032ec5898a88c22a7ac79d1d440370bce5af987'
}


class EnsembledGPT2Config(GPT2Config):
    def __init__(
            self,
            num_models=2,
            non_linearity=False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_models = num_models
        self.non_linearity = non_linearity


class EnsembledGPT2LMHeadModel(PreTrainedModel):
    config_class = EnsembledGPT2Config

    def __init__(self, config):
        super().__init__(config)

        self.model = torch.nn.ModuleList(
            [GPT2LMHeadModel(config) for _ in range(self.config.num_models)]
        )  # initialize with empty models

        # MSCLAR: simplest is a linear layer, check bounds of this and then also include some non-linearity nn.ReLU()
        assert type(self.config.non_linearity) == bool, "non_linearity must be boolean but it is {}".format(
            type(self.config.non_linearity))
        if self.config.non_linearity:
            hidden_size = len(self.model) * self.config.hidden_size // 5
            self.lm_head = torch.nn.Sequential(
                torch.nn.Linear(len(self.model) * self.config.hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1)
            )
        else:
            self.lm_head = torch.nn.Linear(len(self.model) * self.config.hidden_size, 1)

    def initialize_with_existing_models(self, model_names_list, num_models, identical_models=False):
        print(" >>>>>>>> initialize >>>>>>>>> ")
        if model_names_list:
            model_names = [model_name for model_name in model_names_list.split(',')]
        else:
            if identical_models:
                # for analyses purposes
                model_names = [f"stanford-crfm/caprica-gpt2-small-x81" for i in range(num_models)]
            else:
                assert num_models <= len(ALL_MODELS_SMALL)
                model_names = ALL_MODELS_SMALL[:num_models]

        print(f">>>> loading the following models: {model_names} ")

        assert self.config.num_models == len(model_names), "Number of GPT2s models must match"

        for idx, model_name in enumerate(model_names):
            print(f" *** setting parameters of model {model_name}")
            revision = mapping_revision_from_model[model_name]
            model1 = GPT2LMHeadModel.from_pretrained(model_name, revision=revision)
            self.model[idx].load_state_dict(model1.state_dict())

            # TODO: make this a parameter for our evaluations
            # for p in self.model[idx].parameters():
            #     p.requires_grad = False

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        combined_logits = []
        combined_loss = []
        outputs = [
            self.model[i](
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ) for i in range(len(self.model))
        ]

        for o in outputs:
            combined_logits.append(o.logits)
            if o.loss:
                combined_loss.append(o.loss)

        combined_logits = torch.stack(combined_logits, dim=-1).sum(dim=-1) * (1 / len(combined_logits))
        if len(combined_loss) > 0:
            combined_loss = sum(combined_loss)
        # hidden_states = torch.cat([out.last_hidden_state[:, 0, :] for out in outputs], dim=1)  # [batch_size, 2 * hidden_size]
        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.transformer.first_device)
        #     hidden_states = hidden_states.to(self.lm_head.weight.device)
        # lm_logits = self.lm_head(hidden_states)

        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # if not return_dict:
        #     output = (lm_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=combined_loss,
            logits=combined_logits,
            # past_key_values=transformer_outputs.past_key_values,
            # hidden_states=transformer_outputs.hidden_states,
            # attentions=transformer_outputs.attentions,
            # cross_attentions=transformer_outputs.cross_attentions,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        for i in range(len(self.model)):
            self.model[i].resize_token_embeddings(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_models', type=int, default=2)
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--non_linearity', action='store_true')
    parser.add_argument('--model_names_list', type=str, default='')
    parser.add_argument('--freeze_models', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()
    print(json.dumps(vars(args)))

    revision = mapping_revision_from_model[ALL_MODELS_SMALL[0]]
    tokenizer = GPT2Tokenizer.from_pretrained(ALL_MODELS_SMALL[0], revision=revision)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpt_config = GPT2LMHeadModel.from_pretrained(ALL_MODELS_SMALL[0]).config

    path = f'ensembed_gpt_{args.num_models}'
    if args.load_model:
        # initialize an empty GPT2 ensemble
        non_linearity = True if args.non_linearity else False
        config = EnsembledGPT2Config(num_models=args.num_models, non_linearity=non_linearity, **gpt_config.to_dict())
        model = EnsembledGPT2LMHeadModel(config)

        # assign the pre-trained weights
        model.initialize_with_existing_models(args.model_names_list, args.num_models)

        model.save_pretrained(path)

        print(" ====> after saving ....")
        for i in range(len(model.model)):
            print(model.model[i].transformer.wpe)
        print('')

    else:
        model = EnsembledGPT2LMHeadModel.from_pretrained(path)

        print("  ====>  after loading ....")
        for i in range(len(model.model)):
            print(model.model[i].transformer.wpe)
        print('')

    input_ids = tokenizer("The best vacation place is", return_tensors="pt")
    # labels = tokenizer(" a nice coffeeshop in Seattle.", return_tensors="pt")

    outputs = model.forward(
        input_ids=input_ids['input_ids'],
        labels=input_ids['input_ids']
    )
    res = model.generate(input_ids['input_ids'])
    print(tokenizer.batch_decode(res, skip_special_tokens=True))
