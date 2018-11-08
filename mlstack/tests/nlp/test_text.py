# coding=utf-8
"""
    Test for Text
    Created by xueyintao on 2018/11/8.
"""
from mlstack.nlp.text import *

def test_word_freq():

    # test text freq
    tokens = ['a','b','c','b','b','a']
    wfreq = word_freq(tokens)
    print(wfreq)

    sfreq = sorted_word_freq(tokens)
    print(sfreq)

    top2 = sfreq[:2]
    words = []
    for s in top2:
        words.append(s[0])
    print("words:", words)

    topN = top_word_freq(tokens, 1)
    print("topN:", topN)


