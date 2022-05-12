import argparse
from transformers import AutoModelForMultipleChoice
from datasets import load_dataset, load_metric
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, BertModel
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from multiple_choice_ensemble import EnsembledBertForMultipleChoice, EnsembledBertConfig
import datasets


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        if "label" in features[0].keys():
            label_name = "label"
        elif "labels" in features[0].keys():
            label_name = "labels"
        elif "answerKey" in features[0].keys():
            label_name = "answerKey"
        else:
            raise ValueError("No label found in features: {}".format(features[0].keys()))

        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor([int(x) for x in labels], dtype=torch.int64)
        return batch


def main(
        model_name=str,
        train_size=int,
        dev_size=int,
        test_size=int,
        batch_size=int,
        epochs=int,
        save_dir=str,
        learning_rate=float,
        non_linearity=bool,
        num_models=int,
        dataset_name= str,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


    def preprocess_function_arc(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = [[context] * 5 for context in examples["question"]]
        # Grab all second sentences possible for each context.

        second_sentences = []
        for x in examples["choices"]:
            candidates = [f"({l}) {t} " for l, t in zip(x['label'], x['text'])]
            if len(candidates) == 3:
                candidates.append("(D) - ")
            if len(candidates) == 4:
                candidates.append("(E) - ")
            assert len(candidates) == 5, f"{candidates}"
            second_sentences.append(candidates)

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        # Un-flatten
        features = {k: [v[i:i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

        labels = []
        for x in examples["answerKey"]:
            if ord(x) >= ord('A') and ord(x) <= ord('Z'):
                labels.append(ord(x) - ord('A'))
            elif ord(x) >= ord('0') and ord(x) <= ord('9'):
                labels.append(int(x))
            else:
                raise ValueError(f"Invalid label: {x}")
        features["labels"] = labels

        return features

    ending_names = ["ending0", "ending1", "ending2", "ending3"]

    def preprocess_function_swag(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = [[context] * 4 for context in examples["sent1"]]
        # Grab all second sentences possible for each context.
        question_headers = examples["sent2"]
        second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in
                            enumerate(question_headers)]

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        # Un-flatten
        return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    def preprocess_function_hellaswag(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = [[context] * 4 for context in examples["ctx"]]
        # Grab all second sentences possible for each context.
        second_sentences = examples["endings"]
        for x in second_sentences:
            assert len(x) == 4

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        # Un-flatten
        return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    def preprocess_function_boolq(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = []
        second_sentences = []
        for x in examples:
            question = f"{x['question']} - Passage: {x['passage']}"
            first_sentences.append([question, question])
            second_sentences.append([[" (1) no ", " (2) yes "]])

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        # Un-flatten
        return {k: [v[i:i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}

    def preprocess_function_copa(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = []
        second_sentences = []
        for x in examples:
            question = f"{x['premise']} - Choice 1: {x['choice1']} - Choice 2: {x['choice2']}"
            first_sentences.append([question, question])
            second_sentences.append([[" (1) choice 1 ", " (2) choice 2 "]])

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        # Un-flatten
        return {k: [v[i:i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}

    def preprocess_function_mrpc(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = []
        second_sentences = []
        for x in examples:
            input = f"Sent1: {x['sentence1']} - Sent2: {x['sentence2']}"
            first_sentences.append([input, input])
            second_sentences.append([[" (1) Sent1 ", " (2) Sent2 "]])

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        # Un-flatten
        return {k: [v[i:i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}

    def compute_metrics_accuracy(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}



    if dataset_name == "arc_easy":
        datasets = load_dataset("ai2_arc", "ARC-Easy")
        preprocess_function = preprocess_function_arc
        compute_metrics = compute_metrics_accuracy
    elif dataset_name == "arc_hard":
        datasets = load_dataset("ai2_arc", "ARC-Challenge")
        preprocess_function = preprocess_function_arc
        compute_metrics = compute_metrics_accuracy
    elif dataset_name == "swag":
        datasets = load_dataset("swag", "regular")
        preprocess_function = preprocess_function_swag
        compute_metrics = compute_metrics_accuracy
    elif dataset_name == "hellaswag":
        datasets = load_dataset("hellaswag")
        preprocess_function = preprocess_function_hellaswag
        compute_metrics = compute_metrics_accuracy
    elif dataset_name == "mrpc":
        dataset = load_dataset("glue", "mrpc")
        preprocess_function = preprocess_function_mrpc
        compute_metrics = load_metric("glue", "mrpc")
    elif dataset_name == "boolq":
        dataset = load_dataset("super_glue", "boolq")
        preprocess_function = preprocess_function_boolq
        compute_metrics = load_metric("super_glue", "boolq")
    elif dataset_name == "copa":
        dataset = load_dataset("super_glue", "copa")
        preprocess_function = preprocess_function_copa
        compute_metrics = load_metric("super_glue", "copa")
    elif dataset_name == "dream":
        dataset = load_dataset("dream")
        preprocess_function = None
        compute_metrics = compute_metrics_accuracy
        raise NotImplementedError
    elif dataset_name == "record":
        dataset = load_dataset("super_glue", "record")
        preprocess_function = None
        compute_metrics = load_metric("super_glue", "record")
        raise NotImplementedError
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

    datasets.cleanup_cache_files()

    encoded_datasets = datasets.map(preprocess_function, batched=True)

    train_dataset = encoded_datasets["train"].shuffle(seed=42)
    if train_size < 0:
        train_dataset = train_dataset.select(range(train_size))

    dev_dataset = encoded_datasets["validation"].shuffle(seed=42)
    if dev_size < 0:
        dev_dataset = dev_dataset.select(range(dev_size))

    test_dataset = encoded_datasets["test"].shuffle(seed=42)
    if test_size < 0:
        test_dataset = test_dataset.select(range(test_size))


    # model = AutoModelForMultipleChoice.from_pretrained(model_name)
    # non_linearity = True
    # num_models = 2
    bert_config = BertModel.from_pretrained("google/multiberts-seed_0").config
    config = EnsembledBertConfig(num_models=num_models, non_linearity=non_linearity, **bert_config.to_dict())
    model = EnsembledBertForMultipleChoice(config)
    model.initialize_with_existing_berts(model_names_list=None, num_models=num_models)

    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForMultipleChoice(tokenizer)
    )

    trainer.train()
    trainer.save_model(save_dir)
    trainer.save_state()

    import json
    states = open(save_dir + '/trainer_state.json', 'r')
    metrics = json.load(states)
    metrics['model_name'] = model_name
    metrics['learning_rate'] = learning_rate
    metrics['epochs'] = epochs
    metrics['num_models'] = num_models
    metrics['non_linearity'] = non_linearity
    metrics['batch_size'] = batch_size
    metrics['max_eval_accuracy'] = max([x['eval_accuracy'] for x in metrics['log_history'] if 'eval_accuracy' in x])
    metrics['min_eval_loss'] = min([x['eval_loss'] for x in metrics['log_history'] if 'eval_loss' in x])

    m1 = trainer.evaluate(test_dataset, metric_key_prefix="final_test")
    m2 = trainer.evaluate(dev_dataset, metric_key_prefix="final_dev")
    for k, v in m1.items():
        metrics[k] = v
    for k, v in m2.items():
        metrics[k] = v

    print("saving the metrics")
    print(json.dumps(metrics, indent=4, sort_keys=True))

    out = open(save_dir + '/metrics.json', 'w')
    json.dump(metrics, out)
    out.close()


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--train_size")
    parser.add_argument("--dev_size")
    parser.add_argument("--test_size")
    parser.add_argument("--batch_size")
    parser.add_argument("--epochs")
    parser.add_argument("--save_dir")
    parser.add_argument("--learning_rate")
    parser.add_argument("--non_linearity")
    parser.add_argument("--num_models")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    assert args.non_linearity in ['True', 'true', 'False',
                                  'false'], f"{args.non_linearity} - {type(args.non_linearity)}"
    args.non_linearity = args.non_linearity in ['True', 'true']

    main(
        args.model,
        train_size=int(args.train_size),
        dev_size=int(args.dev_size),
        test_size=int(args.test_size),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        save_dir=args.save_dir,
        learning_rate=float(args.learning_rate),
        non_linearity=args.non_linearity,
        num_models=int(args.num_models),
        dataset_name=args.dataset
    )
