## these experiments are run with transformers==4.18.0

from transformers import BertModel, BertConfig
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer

configuration = BertConfig()
model = BertModel(configuration)
configuration = model.config

squad = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

from transformers.data.data_collator import DefaultDataCollator

data_collator = DefaultDataCollator()

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

squad_metric = load_metric("squad")

# Daniel: I am not sure if this is the right way to do this.
def compute_metrics(eval_pred):
    logits, references = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return squad_metric.compute(predictions=predictions, references=references)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"].select(range(500)),  # choose a small subset for faster training
    eval_dataset=tokenized_squad["validation"].select(range(500)), # choose a small subset for faster evaluation
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

