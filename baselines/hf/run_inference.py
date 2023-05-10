import os
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from sent_splitter import SentSplitter
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

model_name_or_path = "models/mt5_en"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
config = AutoConfig.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config).to("cuda")

text_column, summary_column, prefix = "text", "headline", ""
data_files = {
    "test": "test/test_en.json",
}
raw_datasets = load_dataset("json", data_files=data_files, cache_dir=os.environ["SLURM_TMPDIR"])

max_source_length = 512
max_target_length = 64
padding = False

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

ssplitter = SentSplitter()
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(ssplitter.split(pred)) for pred in preds]
    labels = ["\n".join(ssplitter.split(label)) for label in labels]
    return preds, labels

processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=["text", "headline", "langCode"],
    desc="Running tokenizer on dataset",
)
test_dataset = processed_datasets["test"]

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=None,
)

ssplitter = SentSplitter()
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(ssplitter.split(pred)) for pred in preds]
    labels = ["\n".join(ssplitter.split(label)) for label in labels]

    return preds, labels

test_dataloader = DataLoader(
    test_dataset,
    num_workers=2,
    collate_fn=data_collator,
    batch_size=24,
)

metric = evaluate.load("rouge")
def _evaluate():
    model.eval()
    gen_kwargs = {
        "max_length": max_target_length,
        "num_beams": 4,
    }
    progress_bar = tqdm(test_dataloader, desc="Evaluating")
    for batch in progress_bar:
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                **gen_kwargs,
            ).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
    result = metric.compute(tokenizer=lambda x: x.split(), use_stemmer=False)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result

print(_evaluate())