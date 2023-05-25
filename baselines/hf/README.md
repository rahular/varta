## Finetuning models

- `run_sum.py` is the main file for finetuning models on headline generation and abstractive summarization experiments for Varta and Xl-sum datasets.
- `configs` folder contains the configurations for training, model, and data.
- `scripts` folder contains the evaluation scripts to evaluate the model performance.

*Note*: all our experiments were run on pytorch compiled with Cuda 11.7

To run the experiments, activate the virtual environment and run the following line with the corresponding config file under `configs` folder.

```
python -m torch.distributed.launch --nproc_per_node [# of gpus] run_sum.py [config_file.json]
```
