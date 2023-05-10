"""pegasus dataset."""

import json
import tensorflow_datasets as tfds

# TODO(pegasus): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(pegasus): BibTeX citation
_CITATION = """
"""

class PegasusConfig(tfds.core.BuilderConfig):
  """BuilderConfig for varta."""

  def __init__(self, *, language=None, **kwargs):
    """BuilderConfig for varta.

    Args:
      language: string, the language code for the varta dump to use.
      **kwargs: keyword arguments forwarded to super.
    """
    super(PegasusConfig, self).__init__(
        name=f"{language}",
        description=f"Dataset for {language}",
        **kwargs)
    self.language = language

class Pegasus(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for pegasus dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
    PegasusConfig(language=lang)
    for lang in ['ar', 'as', 'bh', 'bn', 'en', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te', 'ur']
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(pegasus): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'inputs': tfds.features.Text(),
            'targets': tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('inputs', 'targets'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(pegasus): Downloads the data and defines the splits
    train_path = '/path/to/pegasus_train'
    dev_path = '/path/to/pegasus_val'
    test_path = '/path/to/pegasus_test'

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={'filepath': train_path, 'language': self._builder_config.language, 'split': 'train'},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={'filepath': dev_path, 'language': self._builder_config.language, 'split': 'val'},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={'filepath': test_path, 'language': self._builder_config.language, 'split': 'test'},
        )
    ]

  def _generate_examples(self, filepath, language, split):
    """Yields examples."""
    # TODO(pegasus): Yields (key, example) tuples from the dataset
    with open(f'{filepath}/{split}_{language}.json', 'r') as f:
      lines = f.readlines()
      for i, row in enumerate(lines):
        inputs, targets = json.loads(row)['inputs'].strip(), json.loads(row)['targets'].strip()
        yield i, {
            'inputs': inputs,
            'targets': targets,
        }