# coding=utf-8
"""
   Text utils
    Created by xueyintao on 2018/11/8.
"""

def word_freq(tokens):
    """
    count token frequence
    :param tokens: tokens array, like ['a','b','c','b','b','a']
    :return: dict of word and freq like {'b': 3, 'c': 1, 'a': 2}
    """
    word_dict = dict()
    for token in tokens:
        if token in word_dict.keys():
            cnt = word_dict.get(token)
            word_dict[token] = cnt + 1
        else:
            word_dict[token] = 1
    return word_dict


def sorted_word_freq(tokens, sort='desc'):
    """
    count token frequence and sort
    :param tokens: tokens array, like ['a','b','c','b','b','a']
    :param sort: 'desc' or 'asc'
    :return: sorted dict of word and freq
    """
    freq = word_freq(tokens)
    reverse = True
    if sort != 'desc':
        reverse = False
    sorted_value = sorted(freq.items(), key=lambda kv: kv[1], reverse=reverse)
    return sorted_value


def top_word_freq(tokens, topN=100, sort='desc'):
    """
    get top N tokens by frequence
    :param tokens: tokens array, like ['a','b','c','b','b','a']
    :param topN: top N
    :param sort: 'desc' or 'asc'
    :return: array of top N words
    """
    sorted_value = sorted_word_freq(tokens, sort)
    top_words = []
    for s in sorted_value[:topN]:
        top_words.append(s[0])
    return top_words
