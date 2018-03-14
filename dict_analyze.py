#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='transform plain text to dictionary.')
parser.add_argument('input_file', help='input file name')
#parser.add_argument('tsv_output_path', type=int, help='the directory for tsv output')
args = parser.parse_args()
i_desc_file = args.input_file

# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from collections import Counter
from nltk.util import ngrams
from nltk.util import everygrams
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import pickle
import csv
import copy

import re

split_mark = "#"  # Split mark for output 覺得使用 space 比較好
filtered_chars = ""  # Characters to ignore
split_chars = "．_-,?…《》，、。？！；：“”‘’'\n\r=—()（）【】■█★✪『』［］[]〔〕"  # Characters representing split mark
# . - 發現是有意義的字元

def str_replace(string, str_from, str_to=""):
    """
    Replace str_from with str_to in string.
    """
    return str_to.join(string.split(str_from))


def str_replace_re(string, str_from, str_to=""):
    """
    Replace str_from with str_to in string.
    str_from can be an re-expression.
    """
    return re.sub(str_from, str_to, string)
    return string

def preprocessing(string):
    """
    Preprocess string.
    :return: processed string
    """
    string = str_replace_re(string, "正文 第.{1,5}回")

    for char in filtered_chars:
        string = str_replace(string, char)

    for char in split_chars:
        string = str_replace(string, char, split_mark)
        string = str_replace(string, "&nbsp;", split_mark)
        string = re.sub(' {2,}',split_mark,string)
        #string = str_replace(string, " ", '_') 

    # Remove consecutive split marks
    while split_mark + split_mark in string:
        string = str_replace(string, split_mark + split_mark, split_mark)

    return string

# -*- coding: utf-8 -*-
import itertools, unicodedata

#split_chars = "…《》，、。？！；：“”‘’'\n\r-=—()（）.【】■★ 『』［］[]〔〕"  # Characters representing split mark
#known_punctuation = u'／（）、，。：「」…。『』！？《》“”；’ ‘【】·〔〕'
known_punctuation = ['.', '/', '-', '／']

def isConnection(char):
    return char in known_punctuation

# ref: https://github.com/hermanschaaf/mafan/blob/master/mafan/text.py 
def has_punctuation(word):
    u"""
    Check if a string has any of the common Chinese punctuation marks.
    """
    if re.search(r'[%s]' % known_punctuation, word) is not None:
        return True
    else:
        return False

def split_words(s):
    # This is a closure for key(), encapsulated in an array to work around
    # 2.x's lack of the nonlocal keyword.
    sequence = [0x10000000, 0, False] # seq, last_key, is_using_concat_string
    #sequence = [0x10000000, 0] 
    def key(part):
        #print(part)
        val = ord(part)
        if part.isspace():
            return 0
        if isConnection(part): # Connect符號. 合併數字 9.7 and 2.0 
            return 1 # sequence[1]
        if part.isdigit(): # 數字
            return 9
        # This is incorrect, but serves this example; finding a more
        # accurate categorization of characters is up to the user.
        asian = unicodedata.category(part) == "Lo"
        if asian:
            # Never group asian characters, by returning a unique value for each one.
            sequence[0] += 1
            return sequence[0]

        return 2

    result = []
    for key, group in itertools.groupby(s, key):
        # Discard groups of whitespace.
        #print(key, group)        
        if key == 0: continue
        sequence[1] = key
        str = "".join(group)
        result.append(str)

    return result

# -*- coding: utf-8 -*-
import itertools, unicodedata

#split_chars = "…《》，、。？！；：“”‘’'\n\r-=—()（）.【】■★ 『』［］[]〔〕"  # Characters representing split mark
#known_punctuation = u'／（）、，。：「」…。『』！？《》“”；’ ‘【】·〔〕'
known_punctuation = ['.', '/', '-', '／']

def isConnection(char):
    return char in known_punctuation


# ref: https://github.com/hermanschaaf/mafan/blob/master/mafan/text.py 
def has_punctuation(word):
    u"""
    Check if a string has any of the common Chinese punctuation marks.
    """
    if re.search(r'[%s]' % known_punctuation, word) is not None:
        return True
    else:
        return False

def split_words(s):
    # This is a closure for key(), encapsulated in an array to work around
    # 2.x's lack of the nonlocal keyword.
    sequence = [0x10000000, 0, False] # seq, last_key, is_using_concat_string
    #sequence = [0x10000000, 0] 
    def key(part):
        #print(part)
        val = ord(part)
        if part.isspace():
            return 0
        if part.isdigit(): # 數字
            return 9
        # This is incorrect, but serves this example; finding a more
        # accurate categorization of characters is up to the user.
        asian = unicodedata.category(part) == "Lo"
        if asian:
            # Never group asian characters, by returning a unique value for each one.
            sequence[0] += 1
            return sequence[0]

        if isConnection(part): # Connect符號. 合併數字 9.7 and 2.0 
            return sequence[1]
        if part.isalnum(): # 數字
            return 8
        return 2

    result = []
    for key, group in itertools.groupby(s, key):
        # Discard groups of whitespace.
        if key == 0: continue
        sequence[1] = key
        str = "".join(group)
        result.append(str)

    return result

print('begin read line from file '+i_desc_file)
input_file = open(i_desc_file, "r", encoding='utf8')  # Open input file
full_txt = input_file.read()
print('original total string length %d' % (len(full_txt)) )
prep_txt = preprocessing(full_txt)

# Counting the words Frequency
max_len_words = 6
occur_of_words = Counter()
# 计算每一个 “words” 的 "出现次数"
print("begin count words occur")
for sentence in prep_txt.split('#'):
    words = split_words(sentence)
    #print(words)
    everygrm = everygrams(words, max_len=max_len_words)
    occur_of_words.update(everygrm)

print('extract number of terms: %s' %(len(occur_of_words)))

print("begin count words length")
occur_of_wordLength = Counter()
# 计算每一个 “words长度” 的 "出现次数"
for i in dict(occur_of_words):
    increase = occur_of_words[i]
    occur_of_wordLength.update( {len(i): increase})

print('count number of terms group by length: %s' %(occur_of_wordLength))

# create pd.DataFrame
df_dict = pd.DataFrame.from_dict(dict(occur_of_words), orient='index')
df_dict.columns=['occur']

def join_lower(words):
    # trim space and transform lower case
    return ''.join([s.strip().lower() for s in words])

df_dict['samp'] = df_dict.index.to_series().map(join_lower)
df_dict['len'] = df_dict.index.to_series().map(len,occur_of_words)
#df_dict.sample(n=10)

print("begin dedup words samp")
# list 大小寫重複的 words
df_dict[df_dict["samp"].duplicated()].sort_values(["len", "samp"]).head(10)
# sum the occurs using group by samp
df_sum_occ_by_samp =  df_dict[['occur', 'samp']].groupby('samp', as_index=False).sum()
# occur 最高者為留下來的 Key Index
df_dict_dedup = df_dict.sort_values(["occur"], ascending=False).drop_duplicates(subset=['len', 'samp'],
                                                                           keep='first', inplace=False)
df_dict_dedup = df_dict_dedup[['len','samp']]
#df_dict2.merge(df_dedupe_samp, left_index=True, right_index=True)
df_dict3 = pd.merge(df_dict_dedup, df_sum_occ_by_samp, how='inner', on='samp', left_index=False, right_index=True)

print("before dedup shape: %s" % str(df_dict.shape))
print("after dedup shape: %s" % str(df_dict3.shape))

def simple_prob(len, occ):
    return occ / count_len[len]

# 『自由度』entropy
dict_groupby_len = df_dict3.groupby(['len'])

import math
import multiprocessing as mp

# 改版提升上面邏輯的速度
def entropy2(ma):
    ma2 = ma / np.sum(ma)
    return np.sum(ma2 * -np.log2(ma2))

def right_entropy2(words, n_words, occur_wi, samp, df_base):
    #print("\tcalc right_entropy2 by %s, %s" % (n_words, samp))
    if n_words >= 6:
        return -1
    df = df_base[ df_base["samp"].str.startswith(samp) ]
    if len(df) == 0:
        return None
    return entropy2(df['occur'].as_matrix())

def left_entropy2(words, n_words, occur_wi, samp, df_base):
    #print("\tcalc left_entropy2 by %s, %s" % (n_words, samp))
    if n_words >= 6:
        return -1
    df = df_base[ df_base["samp"].str.endswith(samp) ]
    if len(df) == 0:
        return None
    return entropy2(df['occur'].as_matrix())

def process_left_entropy(n_words, df_sub, df_base):
    print("begin process_left_entropy len=%s" % (n_words))
    return df_sub.apply(lambda x: left_entropy2(x.name, x.len, x.occur, x.samp, df_base), axis=1)


def process_right_entropy(n_words, df_sub, df_base):
    print("begin process_right_entropy len=%s" % (n_words))
    return df_sub.apply(lambda x: right_entropy2(x.name, x.len, x.occur, x.samp, df_base), axis=1)

res_l_entropy = []
res_r_entropy = []

pool = mp.Pool(processes = 8)


for l in range(2,6):
    df_sub = df_dict3[df_dict3['len'] == l][['len','occur','samp']]
    #print("begin process l=%s, data.shape=%s" % (l, df_sub.shape))
    df_base = dict_groupby_len.get_group(l+1)
    res_l_entropy.append( pool.apply_async(process_left_entropy, args=(l, df_sub, df_base)) )
    res_r_entropy.append( pool.apply_async(process_right_entropy, args=(l, df_sub, df_base)) )

print("begin apply probability...")
# Metric 1: 詞頻
# term 長度的次數 
x = df_dict3[['len','occur']].groupby(["len"]).sum() # sort=False
count_len = x['occur']
df_dict3['prob'] = df_dict3.apply(lambda r: simple_prob(r.len, r.occur), axis=1)

pool.close()
pool.join()

left_entropy_of_words = pd.concat([result.get() for result in res_l_entropy])
right_entropy_of_words = pd.concat([result.get() for result in res_r_entropy])

df_dict3['left_entropy'] = left_entropy_of_words
df_dict3['right_entropy'] = right_entropy_of_words

# save df_dict3
print("save df_dict at %s.pkl ..." % i_desc_file )
df_dict3.to_pickle(i_desc_file + '.pkl')


