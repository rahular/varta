import sys
import multiprocessing
import evaluate
from rouge_tokenizer import tokenize
import six
import collections
import spacy
from indicnlp.tokenize import sentence_tokenize 
import stanza
import json

def _rouge_clean(s):
    return s

def cal_rouge(evaluated_ngrams, reference_ngrams):
    from rouge_score import rouge_scorer
    return rouge_scorer._score_ngrams(reference_ngrams,evaluated_ngrams)

def union(candidates):
    union = collections.Counter()
    for c in candidates:
        union += c
    return union

def split_text(text, lang ,nlp=None, trunc=None):
    """Split text into sentences/words
    Args:
        txt(str): text, as a single str
        trunc(int): if not None, stop splitting text after `trunc` words
                    and ignore sentence containing `trunc`-th word
                    (i.e. each sentence has len <= trunc)
    Returns:
        sents(list): list of sentences (just split the text into different sentences)
        sentences(list): list of sentences (= list of lists of words) - destructive
    """
    # special character removal
    sentences = []
    # keep the original sentence
    sents = []
    if lang == 'en':
        sents = [sent.text for sent in nlp(text).sents]
        sentences = [tokenize(sent,lang='en') for sent in sents]
    
    elif lang =='ur' or lang =='ar':        
        nlp = stanza.Pipeline(lang, processors='tokenize',logging_level='WARN',use_gpu=True)
        sents = [sent.text for sent in nlp(text).sentences]
        sentences = [tokenize(sent) for sent in sents]
    # use sentence splitter
    else:
        sents = sentence_tokenize.sentence_split(text,lang=lang)
        sentences = [tokenize(sent) for sent in sents]

    return sents,sentences

def combination_selection():
    pass

def greedy_selection(doc_sent_list, abstract_sent_list,
                    doc_sent_list_original,abstract_sent_list_original,
                     summary_size, exclusive_ngrams=False):
    """Greedy ext-oracle on lists of sentences
    Args:
        doc_sent_list(list): list of source sentences (itself a list of words)
        abstract_sent_list(list): list of target sentences
                                  (itself a list of words)
        summary_size(int): size of the summary, in sentences
    Returns:
        selected(list): list of selected sentences
    """
    def _get_word_ngrams(n, sentences):
        from rouge_score import rouge_scorer
        return rouge_scorer._create_ngrams(sentences,n)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, sent) for sent in sents]
    reference_1grams = _get_word_ngrams(1, abstract)
    evaluated_2grams = [_get_word_ngrams(2, sent) for sent in sents]
    reference_2grams = _get_word_ngrams(2, abstract)

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = union(candidates_1)
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = union(candidates_2)
            rouge_1 = cal_rouge(candidates_1, reference_1grams).fmeasure
            rouge_2 = cal_rouge(candidates_2, reference_2grams).fmeasure
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected, doc_sent_list_original,abstract_sent_list_original,max_rouge
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected), doc_sent_list_original,abstract_sent_list_original,max_rouge

METHODS = {
    "greedy": greedy_selection,
    "combination": combination_selection,
}

def _clean_line(line):
    return line.strip()


def lead_n(n,sentences):
    '''
    n: number of sentence in the generated output
    sentences: list of senteces
    '''
    return sentences[:n]

def process_example(example):
    (   method,
        src_line,
        tgt_line,
        langCode,
        trunc_src,
        summary_length,
        length_oracle,
        nlp,
        ) = example

    src_line = _clean_line(src_line)
    tgt_line = _clean_line(tgt_line)

    src_sentences_original,src_sentences = split_text(src_line,langCode,nlp=nlp,trunc=trunc_src)
    tgt_sentences_original,tgt_sentences = split_text(tgt_line,langCode,nlp=nlp)

    if length_oracle:
        summary_length = len(tgt_sentences)

    ids, sents_ori,refs_ori,max_rouge = method(src_sentences, tgt_sentences, src_sentences_original,tgt_sentences_original,summary_length)
    lead_1_prediction = lead_n(1,src_sentences_original)
    lead_2_prediction = lead_n(2,src_sentences_original)
    return ids, sents_ori,refs_ori,max_rouge,lead_1_prediction,lead_2_prediction


def from_files(inpath, method,lang,outpath=None,summary_length=None,
               length_oracle=False, trunc_src=None, n_thread=1):
    if method in METHODS:
        method = METHODS[method]
    else:
        raise ValueError("Unknow extoracle method '%s', choices are [%s]"
                         % (method, ", ".join(METHODS.keys())))

    if summary_length is None and not length_oracle:
        raise ValueError(
            "Argument [summary_length, length_oracle] "
            + "cannot be both None/False")
    if summary_length is not None and length_oracle:
        raise ValueError(
            "Arguments [summary_length, length_oracle] are incompatible")

    f = open(inpath,'r',encoding='utf-8')
    # produce a reference file for each language in txt format s.t. 
    # each line represents a reference headline for an article
    # produce a prediction file in txt format for each language for each model
    # where one line contains a prediction
    ref = open(f'{outpath}/{lang}_reference.txt', 'w',encoding = 'utf-8')
    ext_pred = open(f'{outpath}/extoracle_{lang}_prediction.txt', 'w',encoding = 'utf-8')
    lead1_pred = open(f'{outpath}/lead1_{lang}_prediction.txt','w',encoding='utf-8')
    lead2_pred = open(f'{outpath}/lead2_{lang}_prediction.txt','w',encoding='utf-8')

    nlp = None
    if lang =='en':
          spacy.prefer_gpu()
          nlp = spacy.load("en_core_web_sm",exclude=["parser"])
          nlp.enable_pipe("senter")
    
    def example_generator():
        for line in f:
            doc = json.loads(line)
            src_line = doc['text']
            tgt_line = doc['headline']
            langCode = doc['langCode']
            example = (
                method,
                src_line,
                tgt_line,
                langCode,
                trunc_src,
                summary_length,
                length_oracle,
                nlp,
            )
            yield example


    with multiprocessing.Pool(n_thread) as p:
        result_iterator = p.imap(process_example, example_generator())
        metric = evaluate.load('rouge')

        for result in result_iterator:
            ids, sents_ori,references,max_rouge,lead_1_predictions,lead_2_predictions = result
            ####### reference
            references = " ".join(references)
            ref.write(references.replace("\n"," ")+'\n')

            ####### ext-oracle predictions
            predictions = " ".join([sents_ori[i] for i in ids])
            ext_pred.write(predictions.replace("\n"," ")+'\n')

            ####### LEAD-1 LEAD-2 predictions
            lead_1_predictions = " ".join(lead_1_predictions)
            lead1_pred.write(lead_1_predictions.replace("\n"," ")+'\n')
  
            lead_2_predictions = " ".join(lead_2_predictions)
            lead2_pred.write(lead_2_predictions.replace("\n"," ")+'\n')

def main():
    import argparse

    parser = argparse.ArgumentParser("Ext-Oracle Summarization")
    parser.add_argument("inpath",
                        help="Path to the input file in json format\
                         (one example per line, with headline and text)")

    parser.add_argument("outpath",
                        help="""Folder where the output files will be saved""")

    parser.add_argument("-method", "-m", default="greedy",
                        choices=METHODS,
                        help="""Ext-Oracle method (combination is more
                                accurate but takes much longer""")
    parser.add_argument("-language",type=str,help="Language")
    parser.add_argument("-length", "-l", type=int, default=None,
                        help="Length of summaries, in sentences")
    parser.add_argument("-length_oracle", action="store_true",
                        help="Use target summary length")
    parser.add_argument("-trunc", "-t", default=None, type=int,
                        help="Truncate source to <= `trunc` words")
    parser.add_argument("-n_thread", default=1, type=int)
    args = parser.parse_args()

    from_files(args.inpath, args.method,
                         args.language,
                         outpath=args.outpath,
                         summary_length=args.length,
                         length_oracle=args.length_oracle,
                         trunc_src=args.trunc,
                         n_thread=args.n_thread)

if __name__ == "__main__":

    main()



