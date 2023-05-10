import sys
import evaluate
from datasets import load_dataset
from tqdm import tqdm

def get_dataset(lang):
    return load_dataset("ai4bharat/IndicParaphrase", lang, split="test")

def get_predictions(model_dir, lang):
    with open(f"{model_dir}/predictions_{lang}.txt", "r") as f:
        return f.read().splitlines()

def get_targets(model_dir, lang):
    # the naming is unfortunate, but this is the target
    with open(f"{model_dir}/references_{lang}.txt", "r") as f:
        return f.read().splitlines()

def sanity_check(model_dir, lang, ds):
    gold_outputs = ds["target"]
    stored_outputs = get_targets(model_dir, lang)
    for g, s in zip(gold_outputs, stored_outputs):
        print(f"{g}\n{s}\n")

def pad_references(references):
    max_len = max(len(refs) for refs in references)
    return [refs + [""] * (max_len - len(refs)) for refs in references]

def main(debug=False):
    model_dir = sys.argv[1]
    alpha = float(sys.argv[2])
    langs = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
    ibleu = evaluate.load("rahular/ibleu")
    for lang in langs:
        ds = get_dataset(lang)
        inputs = ds["input"]
        references = ds["references"]
        if debug:
            sanity_check(model_dir, lang, ds)
        references = pad_references(references)
        predictions = get_predictions(model_dir, lang)
        assert len(inputs) == len(references) == len(predictions)
        results = ibleu.compute(inputs=inputs, predictions=predictions, references=references, alpha=alpha)
        print(f"{lang}: {results['score']:.4f}")


if __name__ == "__main__":
    main()