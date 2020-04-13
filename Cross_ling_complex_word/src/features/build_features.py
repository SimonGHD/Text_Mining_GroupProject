import re
import collections
from collections import Counter
import pandas as pd
import numpy as np


# file_list = ['News_Dev.tsv','WikiNews_Dev.tsv','Wikipedia_Dev.tsv']

def text2feat(file_list):
    df = pd.read_csv(file_list[0], sep='\t', header=None)

    if len(file_list) > 1:
        for f in file_list:
            df_i = pd.read_csv(f, sep='\t', header=None)
            df.append(df_i, sort=False, ignore_index=True)

    df.columns = ['ID', 'text', 'offsetStart', 'offsetEnd', 'complexPhrase', 'nativeAnn', 'nonnativeAnn', 'nativeMark',
                  'nonnativeMark', 'label', 'prob']
    df = df[~df.complexPhrase.str.contains(r'[\s-]', regex=True)]

    full_text = ' '.join(df['text'].unique())
    text_words = re.split(r'\s+', full_text.lower())

    # dictionary with words and their counts
    C = Counter(text_words)
    # Frequency of words: word counts divided by the number of all words in text
    df['freq'] = df['complexPhrase'].apply(lambda x: C[x.lower()] / len(text_words))

    # Length features
    def countvowels(string):
        num_vowels = 0
        for char in string:
            if char in "aeiouAEIOU":
                num_vowels = num_vowels + 1
        return num_vowels

    def countconsonants(string):
        num_cons = 0
        for char in string:
            if char not in "aeiouAEIOU":
                num_cons = num_cons + 1
        return num_cons

    char_length = []
    vow_length = []
    cons_length = []
    # rep_char = []
    for row in df['complexPhrase']:
        char_length.append(len(row))
        vow_length.append(countvowels(row))
        cons_length.append(countconsonants(row))
        # rep_char.append(collections.Counter(row))

    df['char_length'] = char_length
    df['vow_length'] = vow_length
    df['cons_length'] = cons_length

    avg_word_length = np.mean(df['char_length'])
    df['avg_char_length'] = df['char_length'].apply(lambda x: x / avg_word_length)

    # Number of words in the phrase
    df['words_in_phrase'] = df['text'].apply(lambda x: len(re.split(r'\s+', str(x))))

    # Average length of words in the phrase
    def avg_wlength(phrase):
        words = re.split(r'\s+', phrase)
        w_length = [len(w) for w in words]
        return sum(w_length) / len(words)

    # Average length of words in phrase, normalize by length average of all words
    df['avg_wordlen_phrase'] = df['text'].apply(lambda x: avg_wlength(str(x)) / avg_word_length)
    df = df.reset_index()

    # X: feature matrix
    # y: target vector (labels)
    # w: word vector
    X = df[['freq', 'words_in_phrase', 'avg_wordlen_phrase', 'avg_char_length', 'vow_length', 'cons_length']]
    # X = df[['freq','words_in_phrase','avg_char_length']]
    y = df['label']
    w = df['complexPhrase']
    p = df['prob']

    return X, y, w, p

