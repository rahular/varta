"""varta dataset."""

import json
import tensorflow_datasets as tfds

# TODO(varta): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(varta): BibTeX citation
_CITATION = """
"""

class VartaConfig(tfds.core.BuilderConfig):
  """BuilderConfig for varta."""

  def __init__(self, *, language=None, **kwargs):
    """BuilderConfig for varta.

    Args:
      language: string, the language code for the varta dump to use.
      **kwargs: keyword arguments forwarded to super.
    """
    super(VartaConfig, self).__init__(
        name=f"{language}",
        description=f"Dataset for {language}",
        **kwargs)
    self.language = language

class Varta(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for varta dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
    VartaConfig(language=lang)
    for lang in ['ar', 'as', 'bh', 'bn', 'en', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te', 'ur']
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(varta): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'headline': tfds.features.Text(),
            'text': tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('headline', 'text'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(varta): Downloads the data and defines the splits
    train_path = '/path/to/train/json/folder'
    dev_path = '/path/to/val/json/folder'
    test_path = '/path/to/test/json/folder'

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
    # TODO(varta): Yields (key, example) tuples from the dataset
    with open(f'{filepath}/{split}_{language}.json', 'r') as f:
      lines = f.readlines()
      for i, row in enumerate(lines):
        headline, line = json.loads(row)['headline'].strip(), json.loads(row)['text'].strip()
        yield i, {
            'headline': headline,
            'text': line,
        }