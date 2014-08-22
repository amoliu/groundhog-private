import cPickle
import numpy as np


def get_n_substrings(n, sub_sorted):
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
    subs = {sub:True for sub in bestsubs}
    nexti = max(char_dict.values()) +1
    for sub, p in sub_sorted:
        if sub in subs and sub not in char_dict:
            char_dict[sub] = nexti
            nexti +=1

    return char_dict

def create_sequences(ivocab, char_sub_dict, max_word_len):
    ichar_dict = {v:k for k, v in char_sub_dict.iteritems()}
    
    for i in range(max(ivocab.keys())):
        if i in ivocab:
            wordstr = ivocab[i]
        # Need to put a placeholder if there is a missing word
        else:
            wordstr = ''
        word = []
        for i in range(len(wordstr)):
            for j in range(i+1, len(wordstr)+1):
                if wordstr[i:j] in char_sub_dict: 
                    word.append(char_sub_dict[wordstr[i:j]])
            if len(word) < max_word_len:
            word += [0]*(max_word_len-len(word))
        else:
            word = word[:max_word_len]
        assert len(word) == max_word_len, "word is only %d chars"%len(word)
        words.append(np.asarray(word, dtype=np.int16))
    
    print "Found", max(chars.values()), "characters"
    npwords = np.asarray(words)
    return npwords

if __name__ == '__main__':
    with open('/data/lisatmp3/devincol/data/translation/word_counts/subs.count.pkl') as f: 
        sub_counts = cPickle.load(f)
    with open('/data/lisatmp3/devincol/data/translation/word_counts/subs.sorted.pkl') as f: 
        sub_sorted = cPickle.load(f)
    with open('/data/lisatmp3/devincol/data/translation/vocab.unlimited/char_vocab.en.pkl') as f:
        char_dict = cPickle.load(f)
    with open('/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/ivocab.en.pkl') as f:
        ivocab = cPickle.load(f)

    most_common_subs = get_n_substrings(sub_sorted, 10000)
    best_subs = filter_substrings(most_common_subs, 2)
    char_sub_dict = make_substring_dicionary(best_subs, char_dict, sub_sorted)
    np_sequences = create_sequences(ivocab, char_sub_dict, 40)

    file_name = 'vocab.as_chars_subs.' + str(max(char_sub_dict.values())) + '.en.npy'
    path = '/data/lisatmp3/devincol/data/translation/vocab.unlimited' + file_name
    np.save(path, np_sequences)