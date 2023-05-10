## Configs for Varta Headline Generation Experiments
The experiment includes running with 3 models (mT5, mBERT, T5-varta) in 5 settings (en, hi, latin, dvn, all).

Here we provide the configurations that were used for the 15 Varta Headline Generation experiments reported in the paper.

### Model
We used public checkpoints of mT5, and mBERT, and our pre-trained model varta-t5 available on HuggingFace.

### Data
Specify the data file using `train_file`, `validation_file`, and `test_file` arguments.

### Output Directory
Customized output directory can be specified using `output_dir` argument.

For other arguments in the config file, please check `config_defs.txt` file [here](https://github.com/rahular/varta/blob/main/baselines/hf/configs/config_defs.txt).

Note that rather than trying to squeeze optimal performance from models, the aim of our experiments is to do a detailed study of how different training settings affect model behavior.
