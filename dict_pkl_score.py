#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

parser = argparse.ArgumentParser(description='transform plain text to dictionary.')
parser.add_argument('input_file', help='input file name, pkl format')
parser.add_argument('output_file', help='output file name, tsv format')
#parser.add_argument('tsv_output_path', type=int, help='the directory for tsv output')
args = parser.parse_args()
pkl_fn = args.input_file
dict_fn = args.output_file

def z_normalize(value, mu, sigma):
    if pd.isnull(value): # NaN
        return 0
    if sigma == 0:
        return None
    return (value - mu)/sigma

def export_dict(pkl_fn, dict_fn, theta = 0.5):
    print('begin load pd pkl from file'+pkl_fn)
    df_dict = pd.read_pickle(pkl_fn)
    print('df_dict shape: %s' % str(df_dict.shape))
    l_entropy_mean = {}
    r_entropy_mean = {}
    l_entropy_std = {}
    r_entropy_std = {}
    for len in df_dict['len'].unique():
        print("calc mean/std for len=%s" % len)
        l_entropy_mean[len] = df_dict[df_dict.len == len]['left_entropy'].mean()
        r_entropy_mean[len] = df_dict[df_dict.len == len]['right_entropy'].mean()
        l_entropy_std[len] = df_dict[df_dict.len == len]['left_entropy'].std()
        r_entropy_std[len] = df_dict[df_dict.len == len]['right_entropy'].std()
    # 嘗試n normalize
    df_dict['l_entropy_NZ'] = df_dict.apply( lambda row:
                                            z_normalize(row.left_entropy, l_entropy_mean[row.len], l_entropy_std[row.len])
                                            , axis=1)
    df_dict['r_entropy_NZ'] = df_dict.apply( lambda row:
                                            z_normalize(row.right_entropy, r_entropy_mean[row.len], r_entropy_mean[row.len])
                                            , axis=1)
    df_dict['prob_NZ'] = (df_dict['prob'] - df_dict['prob'].mean())/df_dict['prob'].std()
    df_dict['score'] = np.log(df_dict['l_entropy_NZ'] +1)  + np.log(df_dict['r_entropy_NZ'] +1) + np.log(df_dict['prob_NZ'] +1)
    df_dict2 = df_dict.dropna()[(df_dict.len > 1) & (df_dict["l_entropy_NZ"] > 0) & (df_dict["r_entropy_NZ"] > 0)
                   & (df_dict["prob_NZ"] > 0) ]
    df_final = df_dict2[(df_dict2["score"] > df_dict2["score"].mean() + df_dict2["score"].std()*theta)
                      ].sort_values(["score"], ascending=False)
    print("number of dictionary shape: %s" % str(df_final.shape))
    print("begin write tsv to file " + dict_fn)
    with open( dict_fn, 'w' ) as o:
        for (kw, score) in df_final['score'].iteritems():
            #rint(''.join(kw), score)
            o.write("%s\t%s\n" % (''.join(kw), score))

export_dict(pkl_fn, dict_fn)

