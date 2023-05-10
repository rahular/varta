import os
import re
import json


class SentSplitter:
    '''
    A multilingual sentence splitter mainly written for Indic languages.
    It also works for English and other languages, but not tested for them.
    The code is exactly same as: https://github.com/anoopkunchukuttan/indic_nlp_library/blob/master/indicnlp/tokenize/sentence_tokenize.py,
    but tweaked to be self sufficient and not depend on the transliterate functionality.
    '''
    def __init__(self):
        self.delim_pat = re.compile(r"[\.\?!\|\u0964\u0965]")
        cdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(cdir, "ack_chars.json"), encoding="utf-8") as f:
            self.ack_chars = json.load(f)

    def is_acronym_abbvr(self, char):
        return char in self.ack_chars

    def split(self, text):
        cand_sentences = []
        begin = 0
        text = text.strip()
        for mo in self.delim_pat.finditer(text):
            p1 = mo.start()
            if p1 > 0 and text[p1 - 1].isnumeric():
                continue
            end = p1 + 1
            s = text[begin:end].strip()
            if len(s) > 0:
                cand_sentences.append(s)
            begin = p1 + 1
        s = text[begin:].strip()
        if len(s) > 0:
            cand_sentences.append(s)

        final_sentences = []
        sen_buffer = ""
        bad_state = False
        for sentence in cand_sentences:
            words = sentence.split(" ")
            if len(words) == 1 and sentence[-1] == ".":
                bad_state = True
                sen_buffer = sen_buffer + " " + sentence
            elif sentence[-1] == "." and self.is_acronym_abbvr(words[-1][:-1]):
                if len(sen_buffer) > 0 and not bad_state:
                    final_sentences.append(sen_buffer)
                bad_state = True
                sen_buffer = sentence
            elif bad_state:
                sen_buffer = sen_buffer + " " + sentence
                if len(sen_buffer) > 0:
                    final_sentences.append(sen_buffer)
                sen_buffer = ""
                bad_state = False
            else:
                if len(sen_buffer) > 0:
                    final_sentences.append(sen_buffer)
                sen_buffer = sentence
                bad_state = False
        if len(sen_buffer) > 0:
            final_sentences.append(sen_buffer)

        return final_sentences
