#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='transform plain text to dictionary.')
parser.add_argument('input_file', help='input file name')
#parser.add_argument('tsv_output_path', type=int, help='the directory for tsv output')
args = parser.parse_args()
i_desc_file = args.input_file


# import matplotlib.pyplot as plt
import jieba
import jieba.analyse
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
from collections import defaultdict

known_stops = u'。。…！？\n'
known_sep_punctuation = u',"／│()｜〈〉（）、，。：「」…。『』！？《》“”；’‘【】·〔〕［］★◆◢✪．○≡\n\t'

re_split_sentences = re.compile(r"[%s]+" % known_sep_punctuation)

def split_sentences(text):
    u"""
    Split Chinese text into a list of sentences, separated by punctuation.
    >>> sentence = u"你的電子郵件信箱「爆」了！無法寄信給你。我知道，我正在刪除信件中。"
    >>> print('_'.join(split_sentences(text=sentence)))
    你的電子郵件信箱「爆」了_無法寄信給你_我知道，我正在刪除信件中
    """
    return filter(None, re.split(re_split_sentences, text))

def cut_words(doc):
    #line = re.sub(u"[\s+\.\!\/_,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕【】-]+", " ", line)
    #line = re.sub(u"[,$%^*()+\"\']+|[！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕【】★◆]+", " ", line)
    terms = []
    for sent in split_sentences(doc):
        # print(sent)
        for t in jieba.cut(sent, HMM=True): ## cut_All is True
            terms.append(t)
    return filter(None, terms)

known_connected_punctuation = ['.','/','-','／','#','&','x','+','_','"','~','`','*',':','\'','@',' ']
import itertools

def isConnected(char):
    return char in known_connected_punctuation

def ngrams_plus(words, n=3):
    if n <= 0:
        return []
    grams = [words[i:i + n] for i in range(len(words) - n)]
    for g in grams:
        if len(g[0]) > 0 and len(g[-1]) > 0 and isConnected(g[0]) == False and isConnected(g[-1]) == False:
            yield tuple(g)

def everygram_plus(words, max_len=6):
    for n_i in range(max_len+1):
        for t in ngrams_plus(words, n_i):
            yield t

class my_sentence_stream(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname, mode="r", encoding="utf-8"):
            yield [t for t in cut_words(line)]

from nltk.probability import ConditionalFreqDist

max_n_words = 5

# NLTK 有提供一個不錯的 Utility Class
cfdist = ConditionalFreqDist()
print('begin read raw text file with item description', i_desc_file)
for words in my_sentence_stream(i_desc_file):
    cfdist[1].update(words)
    for i in range(2,max_n_words+2): # max_n_words +1,because 0-base index, +2, because one-more-word for entropy calculation
        cfdist[i].update(ngrams_plus(words, n=i))

for n in cfdist.keys():
    print('words count n=', n, ', total', cfdist[n].N())

min_occur = 10
mini_bins = 10
theta = 0.3

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

def stand_sigmoid(a):
    z_score =(a - np.mean(a)) / np.std(a)
    return sigmoid_array(z_score)

def entropy6(obs, pk, qk=1):
    obs_rate = obs / np.sum(obs)
    return -np.sum( obs_rate * np.log2(pk/qk))

def get_entropy(bins, prob_base):
    """ bins: dict(), {next_word, occurs} """
    # print("\tget entropy by %s, %s" % (kw, occur))
    if len(bins) == 0: # 当找不到任何前/后字的时候 entropy 要算多少呢？
        return prob_base
    return entropy6( np.array([val['occur'] for val in bins.values()], dtype=float), np.array([val['freq'] for val in bins.values()], dtype=float), prob_base)

def process_ngrams(n_words, cfdist ):
    """ for multiprocess version """
    print("begin process ngrams n=%s" % (n_words))
    df_dict = pd.DataFrame.from_dict(dict(cfdist[n_words]), orient='index')
    df_dict.columns=['occur']
    print('dict dataframe shape', df_dict.shape)
    
    df_dict = df_dict[df_dict.occur > min_occur]
    # pre aggreate context words group by , improve performance
    agg_EndsWords = defaultdict(dict) 
    agg_StartsWords = defaultdict(dict)    
    for t,ti in cfdist[n_words+1].items():
        #print(j[:-1], j[-1], nj)
        agg_EndsWords[t[1:]][t[0]] = { 'occur': ti, 'freq': cfdist[n_words+1].freq(t) }
        agg_StartsWords[t[:-1]][t[-1]] = { 'occur': ti, 'freq': cfdist[n_words+1].freq(t) }  
    
    df_dict['prob'] = df_dict.index.map(lambda kw: cfdist[n_words].freq(kw)) 
    df_dict['left_entropy'] = df_dict.apply(lambda r: get_entropy( agg_EndsWords[r.name], r.prob), axis=1)
    df_dict['right_entropy'] = df_dict.apply(lambda r: get_entropy( agg_StartsWords[r.name], r.prob), axis=1)
    
    #df_dict['left_entropy'].fillna( df_dict['left_entropy'].median() , inplace=True)
    #df_dict['right_entropy'].fillna( df_dict['right_entropy'].median() , inplace=True)
    df_dict['left_entropy'].fillna( df_dict['prob'] , inplace=True)
    df_dict['right_entropy'].fillna( df_dict['prob'] , inplace=True)
    
    df_dict['prob_z_sigmoid'] = stand_sigmoid(df_dict['prob'].values) 
    df_dict['l_entropy_z_sigmoid'] = stand_sigmoid(df_dict['left_entropy'].values) 
    df_dict['r_entropy_z_sigmoid'] = stand_sigmoid(df_dict['right_entropy'].values) 
    df_dict['score'] = df_dict['prob_z_sigmoid'] + df_dict['l_entropy_z_sigmoid'] + df_dict['r_entropy_z_sigmoid']
    
    return df_dict[(df_dict["prob_z_sigmoid"] > 0.4) & (df_dict["l_entropy_z_sigmoid"] > 0.6) & (df_dict["r_entropy_z_sigmoid"] > 0.6)]


import multiprocessing as mp

async_result = {} # {n_words, ApplyAsyncResult(pd.dataframe)}
pool = mp.Pool(processes = max_n_words-1)

for n_words in range(2,max_n_words+1):
    #print("begin process l=%s, data.shape=%s" % (n_words, df_sub.shape))
    async_result[n_words] = pool.apply_async(process_ngrams, args=(n_words, cfdist))

pool.close()
pool.join()

df_dicts = {}
# retrieve async result object(pd.dataframe)
for (n_words, result) in async_result.items():
    df_dicts[n_words] = result.get()
    print("n_words=", n_words, "dictionary shape", df_dicts[n_words].shape)

df_all = pd.concat(df_dicts.values())

dict_fn = i_desc_file + ".dict"
print("output dictionary file ", dict_fn)
with open( dict_fn, 'w' ) as o:
    for (kw, score) in df_all.sort_values(["score"], ascending=False)['score'].iteritems():
        #rint(''.join(kw), score)
        o.write("%s\t%s\n" % (''.join(kw), score))