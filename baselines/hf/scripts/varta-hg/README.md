# Evaluation scripts for varta headline generation experiments

We use `xl-sum`'s implementation for ROUGE scoring, please [set up](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring) the environment correctly before running the evaluation scripts.

Note that 
- `TEST_DIR` specifies the directory for the test data files
- `MODEL_DIR` is the directory for a specific experiment (for example, `dvn` experiment with mT5 model), where the best checkpoint is used to evaluate the test data.
- Best checkpoint may vary in each run, please change the checkpoint number in `--model_name_or_path` to the best checkpoint that you get.