import os
import sys
import json
import evaluate
import numpy as np

from tqdm import tqdm

sys.path.insert(0, '..')
from sent_splitter import SentSplitter
from joblib import Parallel, delayed

INPUT_DIR = "/path/to/input/json/files"
OUTPUT_DIR = "/path/to/output/pegasus/files"
lang_files = [
    "train_ar.json",
    "train_bh.json",
    "train_en.json",
    "train_hi.json",
    "train_ml.json",
    "train_ne.json",
    "train_pa.json",
    "train_te.json",
    "train_as.json",
    "train_bn.json",
    "train_gu.json",
    "train_kn.json",
    "train_mr.json",
    "train_or.json",
    "train_ta.json",
    "train_ur.json",
]

metric = evaluate.load("rouge")
ssplitter = SentSplitter()


def get_gap_sents(text, fraction=0.2):
    sents = ssplitter.split(text)
    p = [sent + "\n" for sent in sents]
    r = [text] * len(p)
    result = metric.compute(
        predictions=p,
        references=r,
        rouge_types=["rouge1"],
        tokenizer=lambda x: x.split(),
        use_stemmer=False,
        use_aggregator=False,
    )
    ids = np.argsort(result["rouge1"])[::-1]
    fraction = int(np.ceil(len(sents) * fraction))
    gap_sents = [sents[i] for i in ids[:fraction]]
    for idx in ids[:fraction]:
        sents[idx] = "<extra_id_99>"
    return {
        "inputs": " ".join(sents),
        "targets": " ".join(gap_sents)
    }

def cal_rouge(prediction, reference):
    evaluated_ngrams = set(prediction.split())
    reference_ngrams = set(reference.split())
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count
    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def get_gap_sents_fast(text, fraction=0.2):
    text = json.loads(text)["text"].strip()
    reference = text
    predictions = ssplitter.split(reference)
    scores = map(cal_rouge, predictions, [reference] * len(predictions))
    scores = np.array([score["f"] for score in scores])
    ids = np.argsort(scores)[::-1]
    fraction = int(np.ceil(len(predictions) * fraction))
    gap_sents = [predictions[i] for i in np.sort(ids[:fraction])]
    for idx in ids[:fraction]:
        predictions[idx] = "<extra_id_99>"
    return {
        "inputs": " ".join(predictions),
        "targets": " ".join(gap_sents)
    }

def make_one_lang(lang_file):
    infile = open(os.path.join(INPUT_DIR, lang_file), "r")
    outfile = open(os.path.join(OUTPUT_DIR, lang_file), "w")
    print(f"Processing {lang_file}...")

    out_lines = Parallel(n_jobs=64)(
        delayed(get_gap_sents_fast)(line)
        for line in tqdm(infile)
    )
    print(len(out_lines))

    for line in out_lines:
        outfile.write(json.dumps(line, ensure_ascii=False) + "\n")

    outfile.close()
    infile.close()


def main():
    for lang_file in lang_files:
        make_one_lang(lang_file)


if __name__ == "__main__":
    main()