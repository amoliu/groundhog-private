#!/usr/bin/env python

import cPickle
import numpy as np
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def get_n_grams(sub_counts, n):
    return Counter(dict(filter(lambda (k, v) : len(k) == n, sub_counts.items())))

def get_n_substrings(sub_sorted, n):
    """ Returns the n most common substrings """
    isub_sorted = {v:k for k,v in sub_sorted.iteritems()}
    top = [isub_sorted[i] for i in range(n)]
    return top

def sort_by_len(strings):
    return sorted(strings, key=lambda t: len(t[0]), reverse=True)

def filter_substrings(substrings, cutoff):
    """
    Returns the subset of substrings for which every string satisfies the following:
    ss is in the set if there exists no string s in the set that is a superstring
    of ss and for which P(s)/P(ss) <= cutoff.
    """
    by_len = sort_by_len(substrings)
    keeps = []
    bads = {}
    for i in range(len(by_len)):
        s, p = by_len[i]
        good = s not in bads
        sub_s = [s[i:j] for i in range(len(s)) for j in range(i+1, len(s)+1)]
        for ss in sub_s:
            pp = sub_counts[ss]
            if p/pp <= cutoff:
                bads[ss] = True
        if good:
            keeps.append(s)

    return keeps

def make_substring_dicionary(best_substrings, char_dict, sub_sorted):
    subs = {sub:True for sub in best_substrings}
    nexti = max(char_dict.values()) +1
    for sub, p in sub_sorted:
        if sub in subs and sub not in char_dict:
            char_dict[sub] = nexti
            nexti +=1

    return char_dict

def create_sequences(ivocab, char_sub_dict, max_word_len):
    sub_char_dict = {v : k for k, v in char_sub_dict.items()}

    words = []
    for i in range(max(ivocab.keys())):
        if i in ivocab:
            wordstr = ivocab[i]
        # Need to put a placeholder if there is a missing word
        else:
            wordstr = ''

        # Find all known pieces
        word = []
        if 0:
            for i in range(len(wordstr)):
                for j in range(i+1, len(wordstr)+1):
                    if wordstr[i:j] in char_sub_dict:
                        word.append(char_sub_dict[wordstr[i:j]])
        else:
            for j in range(len(wordstr)+1):
                longest_piece = None
                for i in reversed(range(j)):
                    if wordstr[i:j] in char_sub_dict:
                        longest_piece = char_sub_dict.get(wordstr[i:j])
                if longest_piece:
                    word.append(longest_piece)

        logging.debug("{} --> {}".format(wordstr,
            map(lambda x : sub_char_dict[x], word)))

        # Zero padding
        if len(word) < max_word_len:
            word += [0]*(max_word_len-len(word))
        else:
            word = word[:max_word_len]
        assert len(word) == max_word_len, "word is only %d chars"%len(word)

        words.append(np.asarray(word, dtype=np.int16))

    npwords = np.asarray(words)
    return npwords

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    logger.debug("Load stuff")
    with open('/data/lisatmp3/devincol/data/translation/word_counts/subs.count.pkl') as f:
        sub_counts = cPickle.load(f)
    with open('/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/ivocab.en.pkl') as f:
        ivocab = cPickle.load(f)

    logger.debug("Do the job")
    ngrams = [get_n_grams(sub_counts, k) for k in range(1, 6)]
    ngrams[0] = ngrams[0].keys()
    ngrams[1] = ngrams[1].keys()
    ngrams[2] = dict(ngrams[2].most_common(10000)).keys()
    ngrams[3] = dict(ngrams[3].most_common(10000)).keys()
    ngrams[4] = dict(ngrams[4].most_common(10000)).keys()
    sub_list = []
    for ng in ngrams:
        sub_list.extend(ng)
    char_sub_dict = {sub : i for i, sub in enumerate(sub_list)}
    np_sequences = create_sequences(ivocab, char_sub_dict, 40)

    file_name = 'vocab.as_chars_subs.' + str(max(char_sub_dict.values())) + '.en.npy'
    path = './' + file_name
    np.save(path, np_sequences)
