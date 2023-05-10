#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import shutil
import random
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EncoderDecoderModel,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import (
    check_min_version,
    is_offline_mode,
    send_example_telemetry,
)
from transformers.utils.versions import require_version

sys.path.insert(0, '..')
from sent_splitter import SentSplitter


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/summarization/requirements.txt",
)

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
]

SEQ2SEQ_MODELS = [
    "bert-base-multilingual-cased",
    "google/muril-base-cased",
]

SEQ2SEQ_finetuned = [
    # if you want to resume a finetuned seq2seq model, put the path of the checkpoint here
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(
        default=None, metadata={"help": "Language id for summarization."}
    )
    xlsum_hg: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train for summarization or headline generation for xl-sum"
        },
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    lang_code_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the language code (for multilingual summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )

    lang_prefix: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to add language prefix <lang> before every source text."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    early_stopping_patience: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls."
            )
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
    "csebuetnlp/xlsum": ("text", "summary"),
    "ai4bharat/IndicWikiBio": ("serialized_infobox", "summary"),
    "ai4bharat/IndicHeadlineGeneration": ("input", "target"),
    "ai4bharat/IndicSentenceSummarization": ("input", "target"),
    "ai4bharat/IndicParaphrase": ("input", "target"),
    "ai4bharat/IndicQuestionGeneration": ("context", "question"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    config_file_path = None
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config_file_path = sys.argv[1]
    elif (
        len(sys.argv) == 3
        and sys.argv[1].startswith("--local_rank")
        and sys.argv[2].endswith(".json")
    ):
        config_file_path = sys.argv[2]
    if config_file_path is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(config_file_path)
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    random.seed(training_args.data_seed)
    set_seed(training_args.seed)

    # If do_train is False, then manually set do_eval to False.
    # There seems to be a bug in argument parsing, which sets do_eval to True even if explicitly set to False in the config file.
    if not training_args.do_train:
        training_args.do_eval = False

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    datasets.disable_caching()
    datasets_backup_path = os.path.join(
        "cache",
        f"{training_args.output_dir.replace('.', '').replace('/', '_')}_dataset_cache",
    )
    os.makedirs(datasets_backup_path, exist_ok=True)
    cache_path = os.path.join(os.environ["SLURM_TMPDIR"], training_args.output_dir)
    # fmt: off
    dataset_spec_langs = {
        "csebuetnlp/xlsum": ["bengali", "english", "gujarati", "hindi", "marathi", "nepali", "punjabi", "sinhala", "tamil", "telugu", "urdu"],
        "ai4bharat/IndicWikiBio": ["as", "bn", "hi", "kn", "ml", "or", "pa", "ta", "te"],
        "ai4bharat/IndicHeadlineGeneration": ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"],
        "ai4bharat/IndicSentenceSummarization": ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"],
        "ai4bharat/IndicParaphrase": ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"],
        "ai4bharat/IndicQuestionGeneration": ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"],
    }
    # fmt: on
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.dataset_name == "csebuetnlp/xlsum" and data_args.xlsum_hg:
            summarization_name_mapping["csebuetnlp/xlsum"] = ("text", "title")
        if data_args.dataset_name in dataset_spec_langs:
            if data_args.lang not in dataset_spec_langs[data_args.dataset_name] + [
                "all"
            ]:
                raise ValueError(
                    f"Make sure that the language of the dataset is correctly specified. \
                Please pick one among the available languages:\
                {dataset_spec_langs[data_args.dataset_name]}"
                )
            elif data_args.lang == "all":

                def concat_datasets(dname, split):
                    if dname == "ai4bharat/IndicParaphrase" and split == "train":
                        dlangs = dataset_spec_langs[dname].copy()
                        dlangs.remove("as")
                    else:
                        dlangs = dataset_spec_langs[dname]
                    return concatenate_datasets(
                        [
                            load_dataset(
                                dname,
                                lang,
                                cache_dir=cache_path,
                                split=split,
                                use_auth_token=True
                                if model_args.use_auth_token
                                else None,
                            )
                            for lang in dlangs
                        ]
                    )

                raw_datasets = DatasetDict(
                    {
                        "train": concat_datasets(data_args.dataset_name, "train"),
                        "test": concat_datasets(data_args.dataset_name, "test"),
                        "validation": concat_datasets(
                            data_args.dataset_name, "validation"
                        ),
                    }
                )
            else:
                raw_datasets = load_dataset(
                    data_args.dataset_name,
                    data_args.lang,
                    cache_dir=cache_path,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=cache_path,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        if training_args.do_train and data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if training_args.do_eval and data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if training_args.do_predict and data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=cache_path,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=cache_path,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=cache_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.lang_prefix:
        # fmt: off
        varta_langs = ["ar", "as", "bh", "bn", "en", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur"]
        # fmt: on
        special_tokens = [f"<{lang}>" for lang in varta_langs]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
    if model_args.model_name_or_path in SEQ2SEQ_MODELS:
        model_class = EncoderDecoderModel
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_args.model_name_or_path,
            model_args.model_name_or_path,
            tie_encoder_decoder=False,
        )
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        if data_args.lang_prefix:
            _ = tokenizer.add_special_tokens(special_tokens_dict)
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer))
    else:
        model_class = AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=cache_path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if model_args.model_name_or_path in SEQ2SEQ_finetuned:
            if data_args.lang_prefix:
                _ = tokenizer.add_special_tokens(special_tokens_dict)
            model.encoder.resize_token_embeddings(len(tokenizer))
            model.decoder.resize_token_embeddings(len(tokenizer))
        else:
            if data_args.lang_prefix:
                _ = tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))
    cache_path = os.path.join(cache_path, "dataset_cache")

    # if we want to run the script only for predictions, then we need to load the best model first
    if (
        training_args.do_predict
        and not training_args.do_train
        and not training_args.do_eval
    ):
        logger.info(
            f"Loading best model from {training_args.output_dir} for predictions"
        )
        model = model_class.from_pretrained(training_args.output_dir)

    if model.config.decoder_start_token_id is None and isinstance(
        tokenizer, (MBartTokenizer, MBartTokenizerFast)
    ):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
                data_args.lang
            ]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(
                data_args.lang
            )

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token]
            if data_args.forced_bos_token is not None
            else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.lang_code_column is None and data_args.lang_prefix:
        raise ValueError(
            f"Must provide a value for lang_code_column in the case of adding language prefix"
        )
    elif data_args.lang_prefix:
        if data_args.lang_code_column not in column_names:
            raise ValueError(
                f"--lang_code_column' value '{data_args.lang_code_column}' needs to be one of: {', '.join(column_names)}"
            )
        else:
            lang_code_column = data_args.lang_code_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                lang_prefix = (
                    "<" + examples[lang_code_column][i] + ">"
                    if data_args.lang_prefix
                    else ""
                )
                if data_args.dataset_name == "ai4bharat/IndicQuestionGeneration":
                    inputs.append((
                        lang_prefix
                        + prefix
                        + examples[text_column][i],
                        examples["answer"][i]
                    ))
                else:
                    inputs.append(lang_prefix + prefix + examples[text_column][i])
                targets.append(examples[summary_column][i])

        # inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_dataset_sample(dataset, k):
        max_samples = min(len(dataset), k)
        return dataset.select(random.sample(range(len(dataset)), k=max_samples))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            if data_args.dataset_name == None:
                cache_fname = f"train_{data_args.train_file}"
            elif data_args.xlsum_hg:
                dataset_name = data_args.dataset_name.replace("/", "_")
                cache_fname = f"train_{dataset_name}_{data_args.lang}_HG"
            else:
                dataset_name = data_args.dataset_name.replace("/", "_")
                cache_fname = f"train_{dataset_name}_{data_args.lang}"

            train_cache_path = os.path.join(cache_path, cache_fname)
            if not os.path.exists(train_cache_path) and os.path.exists(
                os.path.join(datasets_backup_path, cache_fname)
            ):
                logger.info(f"Copying train dataset features to {train_cache_path}")
                shutil.copytree(
                    os.path.join(datasets_backup_path, cache_fname), train_cache_path
                )
            if os.path.exists(train_cache_path):
                logger.info(
                    f"Loading train dataset features from cached file {train_cache_path}"
                )
                train_dataset = datasets.load_from_disk(train_cache_path)
                if data_args.max_train_samples is not None:
                    train_dataset = get_dataset_sample(
                        train_dataset, data_args.max_train_samples
                    )
            else:
                if data_args.max_train_samples is not None:
                    train_dataset = get_dataset_sample(
                        train_dataset, data_args.max_train_samples
                    )
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                train_dataset.save_to_disk(train_cache_path)
                shutil.copytree(
                    train_cache_path, os.path.join(datasets_backup_path, cache_fname)
                )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            if data_args.dataset_name == None:
                cache_fname = f"eval_{data_args.validation_file}"
            elif data_args.xlsum_hg:
                dataset_name = data_args.dataset_name.replace("/", "_")
                cache_fname = f"eval_{dataset_name}_{data_args.lang}_HG"
            else:
                dataset_name = data_args.dataset_name.replace("/", "_")
                cache_fname = f"eval_{dataset_name}_{data_args.lang}"

            eval_cache_path = os.path.join(cache_path, cache_fname)
            if not os.path.exists(eval_cache_path) and os.path.exists(
                os.path.join(datasets_backup_path, cache_fname)
            ):
                logger.info(f"Copying eval dataset features to {eval_cache_path}")
                shutil.copytree(
                    os.path.join(datasets_backup_path, cache_fname), eval_cache_path
                )
            if os.path.exists(eval_cache_path):
                logger.info(
                    f"Loading eval dataset features from cached file {eval_cache_path}"
                )
                eval_dataset = datasets.load_from_disk(eval_cache_path)
                if data_args.max_eval_samples is not None:
                    eval_dataset = get_dataset_sample(
                        eval_dataset, data_args.max_eval_samples
                    )
            else:
                if data_args.max_eval_samples is not None:
                    eval_dataset = get_dataset_sample(
                        eval_dataset, data_args.max_eval_samples
                    )
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
                eval_dataset.save_to_disk(eval_cache_path)
                shutil.copytree(
                    eval_cache_path, os.path.join(datasets_backup_path, cache_fname)
                )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            if data_args.dataset_name == None:
                cache_fname = f"predict_{data_args.test_file}"
            elif data_args.xlsum_hg:
                dataset_name = data_args.dataset_name.replace("/", "_")
                cache_fname = f"predict_{dataset_name}_{data_args.lang}_HG"
            else:
                dataset_name = data_args.dataset_name.replace("/", "_")
                cache_fname = f"predict_{dataset_name}_{data_args.lang}"

            predict_cache_path = os.path.join(cache_path, cache_fname)
            if not os.path.exists(predict_cache_path) and os.path.exists(
                os.path.join(datasets_backup_path, cache_fname)
            ):
                logger.info(f"Copying predict dataset features to {predict_cache_path}")
                shutil.copytree(
                    os.path.join(datasets_backup_path, cache_fname), predict_cache_path
                )
            if os.path.exists(predict_cache_path):
                logger.info(
                    f"Loading predict dataset features from cached file {predict_cache_path}"
                )
                predict_dataset = datasets.load_from_disk(predict_cache_path)
                if data_args.max_predict_samples is not None:
                    predict_dataset = get_dataset_sample(
                        predict_dataset, data_args.max_predict_samples
                    )
            else:
                if data_args.max_predict_samples is not None:
                    predict_dataset = get_dataset_sample(
                        predict_dataset, data_args.max_predict_samples
                    )
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )
                predict_dataset.save_to_disk(predict_cache_path)
                shutil.copytree(
                    predict_cache_path, os.path.join(datasets_backup_path, cache_fname)
                )
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(
                    len(predict_dataset), data_args.max_predict_samples
                )
                predict_dataset = predict_dataset.select(
                    random.sample(range(len(predict_dataset)), k=max_predict_samples)
                )

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = evaluate.load("rouge")

    ssplitter = SentSplitter()

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(ssplitter.split(pred)) for pred in preds]
        labels = ["\n".join(ssplitter.split(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            tokenizer=lambda x: x.split(),
            use_stemmer=False,
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Initialize our Trainer
    cb_early_stop = EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)
    if data_args.dataset_name == "ai4bharat/IndicParaphrase":
        cb_early_stop = EarlyStoppingCallback(early_stopping_patience=10**6)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.predict_with_generate
        else None,
        callbacks=[cb_early_stop],  # hard-coded
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        def write_to_file(out_file, lines):
            with open(os.path.join(training_args.output_dir, out_file), "w") as writer:
                writer.write("\n".join(lines) + "\n")

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                references = tokenizer.batch_decode(
                    np.where(
                        predict_results.label_ids != -100,
                        predict_results.label_ids,
                        tokenizer.pad_token_id,
                    ),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                references = [ref.strip() for ref in references]
                write_to_file("predictions.txt", predictions)
                write_to_file("references.txt", references)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
