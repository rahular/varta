## Vārta : A Large-Scale Headline-Generation Dataset for Indic Languages

This repository contains the code and other resources for the paper published in the Findings of ACL 2023.

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/ACL-2023%20Findings-blue"></a>
  <a href="https://github.com/rahular/varta/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/Apache%202.0-green">
  </a>
</p>

[**Dataset**](#dataset) |
[**Pretrained Models**](#pretrained-models) |
[**Finetuning**](#finetuning-experiments) |
[**Evaluation**](#evaluation) |
[**Citation**](#citation)

### Dataset
The Vārta dataset is available on the [Huggingface Hub](https://huggingface.co/datasets/rahular/varta). We release train, validation, and test files in JSONL format. Each article object contains: 
- `id:` unique identifier for the artilce on DailyHunt. This id will be used to recreate the dataset.
- `langCode`: ISO 639-1 language code
- `source_url`: the url that points to the article on the website of the original publisher
- `dh_url`: the url that points to the article on DailyHunt
- `id`: unique identifier for the artilce on DailyHunt.
- `url`: the url that points to the article on DailyHunt
- `headline`: headline of the article
- `publication_date`: date of publication
- `text`: main body of the article
- `tags`: main topics related to the article
- `reactions`: user likes, dislikes, etc.
- `source_media`: original publisher name
- `source_url`: the url that points to the article on the website of the original publisher
- `word_count`: number of words in the article
- `langCode`: language of the article

To recreate the dataset, follow this [README file](https://github.com/rahular/varta/tree/main/crawler#README.md).

The `train`, `val`, and `test` folders contain language-specific json files and one aggregated file. However, the `train` folder has multiple aggregated training files for different experiments (you will have to recreate them). The data is structured as follows:
- `train`:
  - `train.json`: large training file
  - `train_small.json`: small training file; training file for the *all* experiments
  - `train_en_1M.json`: training file for the *en* experiments
  - `train_hi_1M.json`: training file for the *hi* experiments
  - `langwise`:
    - `train_<lang>.json`: large language-wise training files
    - `train_<lang>_100k.json`: small language-wise training files
- `test`:
  - `test.json`: aggregated test file
  - `langwise`: 
    - `test_<lang>.json`: language-wise test files
- `val`
  - `val.json`: aggregated validation file
  - `langwise`: 
    - `val_<lang>.json`: language-wise validation files

*Note*: if you don't want to download the whole dataset, and just want one file, you can do something like
```
wget https://huggingface.co/datasets/rahular/varta/raw/main/varta/<split>/langwise/<split>_<lang>.json
```

### Pretrained Models
- We release the Varta-T5 model in multiple formats:
  - For tensorflow, in the t5x format ([t5-small](https://console.cloud.google.com/storage/browser/varta-eu/t5x/varta-t5-small-ckpts), [t5-base](https://console.cloud.google.com/storage/browser/varta-eu/t5x/varta-t5-base-ckpts))
  - For pytorch, as a HF model ([t5-small](https://huggingface.co/rahular/varta-t5-small), [t5-base](https://huggingface.co/rahular/varta-t5))
- We release Varta-BERT only in pytorch as a HF model ([link](https://huggingface.co/rahular/varta-bert))

The code for:
- **Pretraining Varta-T5**: follow the README [here](https://github.com/rahular/varta/tree/main/baselines/vartaT5#readme)
- **Pretraining Varta-BERT** follow the README [here](https://github.com/AI4Bharat/IndicBERT#readme)

### Finetuning Experiments
The code for all finetuning experiments reported in the paper is placed under the [baselines folder](https://github.com/rahular/varta/tree/main/baselines).
- **Extractive Baselines**: follow the README [here](https://github.com/rahular/varta/tree/main/baselines/extractive#readme)
- **Transformer Baselines**: follow the README [here](https://github.com/rahular/varta/blob/main/baselines/hf#readme)

### Evaluation
We use the multilingual variant of ROUGE [implemented](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring) for the xl-sum paper for the evaluations of the headline generation and abstractive summarization tasks in our experiments. 

### Citation
```
@misc{aralikatte2023varta,
      title={V\=arta: A Large-Scale Headline-Generation Dataset for Indic Languages}, 
      author={Rahul Aralikatte and Ziling Cheng and Sumanth Doddapaneni and Jackie Chi Kit Cheung},
      year={2023},
      eprint={2305.05858},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
