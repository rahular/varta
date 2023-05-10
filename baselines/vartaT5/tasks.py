# move this to /t5x/examples/t5/tasks.py

import seqio
import functools
import tensorflow as tf

import t5.data
from t5.data import preprocessors
from t5x.examples.t5 import preprocessors

tf.random.set_seed(42)


TaskRegistry = seqio.TaskRegistry
AUTOTUNE = tf.data.experimental.AUTOTUNE

DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

sentencepiece_model_file = "/home/sumanth/t5-tokenizer/updatedspm128k.model"
vocab = seqio.SentencePieceVocabulary(sentencepiece_model_file)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=vocab, add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=vocab, add_eos=True)
}

LANGS = ['as', 'bh', 'bn', 'en', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te', 'ur']


# pegasus tasks
for lang in LANGS:
    seqio.TaskRegistry.add(
        f"pegasus_gs_{lang}",
        source=seqio.TfdsDataSource(tfds_name=f"pegasus/{lang}:1.0.0"),
        preprocessors=[
            functools.partial(
                preprocessors.rekey, key_map={
                    "inputs": "inputs",
                    "targets": "targets",
                }),
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[])

pegasus = [f"pegasus_gs_{lang}" for lang in LANGS]
seqio.MixtureRegistry.add("pegasus", pegasus, default_rate=DEFAULT_MIX_RATE)


# varta span corruption tasks
for lang in LANGS:
    seqio.TaskRegistry.add(
        f"varta_sc_{lang}",
        source=seqio.TfdsDataSource(tfds_name=f"varta/{lang}:1.0.0"),
        preprocessors=[
            functools.partial(
                preprocessors.rekey, key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            preprocessors.span_corruption,
            seqio.preprocessors.append_eos_after_trim,

        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[])

varta = [f"varta_sc_{lang}" for lang in LANGS]
seqio.MixtureRegistry.add("varta", varta, default_rate=DEFAULT_MIX_RATE)

seqio.MixtureRegistry.add("varta_pegasus", varta + pegasus, default_rate=DEFAULT_MIX_RATE)
