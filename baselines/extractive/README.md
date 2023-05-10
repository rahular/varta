## Extractive baselines

To run the extractive models (Lead 1, Lead 2 and Ext-Oracle), do
```python extoracle_leadn.py [inpath] [outpath] -language=[langcode] -length_oracle```
where:
- `inpath` is the path of the input file (a jsonl file containing `headline`, `text`, and l`angCode`)
- `outpath` is the path of a folder where the output files will be saved

This will produce one file `{lang}_reference.txt` with each line representing the reference from one article, as well as three predictions file `extoracle_{lang}_prediction.txt`, `lead1_{lang}_prediction.txt` and `lead2_{lang}_prediction.txt`, containing the predictions obtained by the respective extractive heuristics.

To evaluate the performance of the extractive models, run `eval_extractive.sh`.
