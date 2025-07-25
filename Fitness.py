import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
import numpy as np
import os
import argparse
import pandas as pd
import random
import re
import skimage
import skimage.io
import scanpy as sc
import anndata as ad
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances
import math

def Norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_pairwise_speed(dist, time_f):
    ##input is distance matrix and pseudotime
    speed_df = np.zeros((len(time_f), len(time_f)))
    speed_df[:] = np.nan
    
    for i in dist.index.values:
        #print(i)
        for j in dist.index.values:
            #print(i,j)
            if i != j :
                v = dist.iloc[i,j]/abs(time_f.loc[i,'dpt_pesudotime'] - time_f.loc[j,'dpt_pesudotime'])
            else:
                v = 0
                
            speed_df[i,j] = v
        
    return speed_df


def main(args):
    if args.input_path is not None:
        if os.path.isdir(args.input_path):
            input_files = [_ for _ in os.listdir(args.input_path) if not _.startswith('.') and _.endswith('.h5ad')]
        else:
            input_files = [args.input_path]
    
    
    #print(input_files)
    results_dict = {}
    
    for f in input_files:
        print(f)
        n_cols, n_rows = int(f.split("_")[1][:-4]), int(f.split("_")[2].split(".")[0][:-4])
        
        adata = ad.read_h5ad(os.path.join(args.input_path,f))
        shannon_index_leiden = entropy(adata.obs['leiden'].value_counts())
        
        col_names = ['norm_col', 'norm_row', 'dpt_pesudotime']
        df = pd.DataFrame(columns=col_names)
        df['norm_col'] = adata.obs.groupby(['leiden','pred_class'])['n_col'].mean()/n_cols
        df['norm_row'] = adata.obs.groupby(['leiden','pred_class'])['n_row'].mean()/n_rows
        df['dpt_pesudotime'] = adata.obs.groupby(['leiden','pred_class'])['dpt_pseudotime'].mean()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=col_names,how='any').reset_index()
    
        if not df.empty:
            coordinates = df[["norm_col","norm_row"]]
            dist =  pd.DataFrame(euclidean_distances(coordinates,coordinates))
            dist = Norm(dist)
            speed_df= get_pairwise_speed(dist, df)
            speed = pd.DataFrame(speed_df).fillna(0).sum().sum()/len(df)
        
            slide_dict = {}
            slide_dict['filename'] = f
            slide_dict['speed'] =  math.log(speed) if speed != 0  else 0
            slide_dict['shannon_index'] =  shannon_index_leiden
        
            results_dict[f] = slide_dict
        
    output_df = pd.DataFrame(results_dict).T    
    output_df['fitness'] = output_df['speed'] * output_df['shannon_index']

    output_df.to_csv(args.output_file,index=False)
     
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Feature Extraction by phikon')

    parser.add_argument('--input_path', default=os.path.join(os.getcwd(), 'results/'), type=str,
                        help="path for extract image features in csv format")
    parser.add_argument('--output_file', default=os.path.join(os.getcwd(), 'output.csv'), type=str,
                        help='output file name')

    args = parser.parse_args()

    main(args)
