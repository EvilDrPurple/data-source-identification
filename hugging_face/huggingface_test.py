from datasets import load_dataset
from datasets import ClassLabel
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from multimodal_transformers.model import AutoModelWithTabular, TabularConfig
from transformers import AutoConfig
from multimodal_transformers.data import load_data
import numpy as np
import pandas as pd
import evaluate


MODEL = "distilbert-base-uncased"
MAX_STEPS = 500

def str2int(label):
    return labels.index(label)

dataset = load_dataset("PDAP/urls-and-headers")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

labels_file = open("labels.txt", "r")
labels = [line.strip() for line in labels_file.readlines()]
labels_file.close()
num_labels = len(labels)
label_col = "label"
train_df["label"] = train_df["label"].apply(str2int)
test_df["label"] = test_df["label"].apply(str2int)

text_cols = ["url", "http_response", "html_title", "meta_description", "h1", "h2", "h3", "h4", "h5", "h6"]
empty_text_values = ['[""]', None, "[]"]
tokenizer = AutoTokenizer.from_pretrained(MODEL)

train_dataset = load_data(train_df, text_cols, tokenizer, label_col, label_list=labels, sep_text_token_str=tokenizer.sep_token, empty_text_values=empty_text_values)
test_dataset = load_data(test_df, text_cols, tokenizer, label_col, label_list=labels, sep_text_token_str=tokenizer.sep_token, empty_text_values=empty_text_values)

config = AutoConfig.from_pretrained(MODEL)
tabular_config = TabularConfig(
    num_labels=num_labels,
    combine_feat_method="text_only",
)
config.tabular_config = tabular_config

model = AutoModelWithTabular.from_pretrained(MODEL, config=config)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./url_classifier",
    logging_dir="./url_classifier/runs",
    overwrite_output_dir=True,
    do_train=True,
    max_steps=MAX_STEPS,
    evaluation_strategy="steps",
    eval_steps=25,
    logging_steps=25,
    #push_to_hub=True
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
  compute_metrics=compute_metrics
)

trainer.train()

'''def tokenize_function(batch):
    for batch_item in batch:
        for x, text in enumerate(batch[batch_item]):
            #print(batch[batch_item])
            if text is None:
                batch[batch_item][x] = "[]"

    #print(batch[0])
    tokens = tokenizer(
        model_args=[batch["url"],
        batch["http_response"],
        batch["html_title"],
        batch["meta_description"],
        batch["h1"],
        batch["h2"],
        batch["h3"],
        batch["h4"],
        batch["h5"],
        batch["h6"]],
        padding="max_length",
        truncation=False,
    )

    tokens["label"] = hh.str2int(batch["label"])
    return tokens


tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Add custom labels to the results
tokenized_datasets = tokenized_datasets.cast_column("label", labels)

# Shuffles the dataset, a smaller range can be selected to speed up training
# Selecting a smaller range may come at the cost of accuracy and may cause errors if some labels end up being excluded from the dataset
train_dataset = tokenized_datasets["train"].shuffle(seed=42)  # .select(range(1000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42)  # .select(range(1000))

classifier = pipeline("text-classification", model=MODEL, framework="pt", tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=28)
training_args = TrainingArguments(
    output_dir="test_trainer_2.0", evaluation_strategy="epoch", max_steps=MAX_STEPS, push_to_hub=True)
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
'''
# These will push a new model version to the Hugging Face Hub upon training completion
#trainer.push_to_hub("PDAP/url-classifier")
#tokenizer.push_to_hub("PDAP/url-classifier")
