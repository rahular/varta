# Varta-T5

### Create Pegasus-style data

This will read the json files and create Pegasus-style data by masking the most important sentence in the summary. The masked data will be stored in json files.
```
python make_pegasus_data.py
```

### Create TFDS dataset

Follow the instructions in the [TFDS README](https://www.tensorflow.org/datasets/add_dataset) to create a TFDS dataset. The dataset should be stored in a GCS bucket.

- move the tfds/varta.py file to respective TFDS code directory
- move the tfds/pegasus.py file to respective TFDS code directory

### Installation

Follow the instructions in the [README](https://github.com/google-research/t5x/blob/main/README.md) to install the t5x dependencies.

### Addtional changes

Update the tokenizer path in [base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/base.gin) with [our](https://console.cloud.google.com/storage/browser/varta-eu/t5x/t5-tokenizer) tokenizer
```
seqio.SentencePieceVocabulary.sentencepiece_model_file = "/path/to/updated/spm/model/updatedspm128k.model"
```

Change the vocab size in [base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/base.gin) to 128k
```
vocab_size = 128128
```

### Add new task and gin files

- move the tasks.py file to /t5x/examples/t5/
- move the pegasus.gin file to /t5x/examples/t5/t5_1_1/examples

### Run the model

```
export TFDS_DATA_DIR=gs://path/to/gcs/bucket/with/tfds/data
export MODEL_DIR=gs://path/to/gcs/bucket
export T5X_DIR=/t5x/dir/in/your/machine

python ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/pegasus.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --alsologtostderr
```