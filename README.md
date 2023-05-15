## Vārta : A Large-Scale Headline-Generation Dataset for Indic Languages

This repository contains the code and other resources for the paper published in the Findings of ACL 2023.

### Dataset
The Vārta dataset is currently available in [this](https://console.cloud.google.com/storage/browser/varta-eu/data-release) bucket. We release train, validation, and test files in JSONL format. Each article object contains: 
  - `id`: unique identifier for the artilce on DailyHunt. This id will be used to recreate the dataset.
  - `langCode`: ISO 639-1 language code
  - `source_url`: the url that points to the article on the website of the original publisher
  - `dh_url`: the url that points to the article on DailyHunt

To recreate the dataset, follow this [README file](https://github.com/rahular/varta/tree/main/crawler#README.md).

The `train`, `val`, and `test` folders contain language-specific json files and one aggregated file. However, the `train` folder has multiple aggregated training files for different experiments. The data is structured as follows:
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

### Pretrained Models
- We release the Varta-T5 model in multiple formats:
  - For tensorflow, in the t5x format ([link](https://console.cloud.google.com/storage/browser/varta-eu/t5x))
  - For pytorch, as a HF model ([link](https://huggingface.co/rahular/varta-t5))
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
