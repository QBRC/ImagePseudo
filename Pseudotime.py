import numpy as np
import os
import argparse
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics.pairwise import euclidean_distances

def Norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def paga(wsi_df):
    x, obs = pd.DataFrame(wsi_df.iloc[:,3:]).astype('float64'), pd.DataFrame(wsi_df.iloc[:,0:3])
    adata = ad.AnnData(x,obs)
    
    print('start paga')
    sc.tl.pca(adata, svd_solver='arpack') ## dimension reduction
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
    sc.tl.leiden(adata,resolution=1) ## clustering
    sc.tl.paga(adata, groups='leiden') ## PAGA for clustering partition

    #start_class = adata.obs['pred_class'].unique().min()
    start_class = 4
    adata.uns['iroot'] = np.flatnonzero(adata.obs['pred_class'] == start_class)[0] ## set predicted G1 class as root node
    
    sc.tl.dpt(adata)  ### DPT to calculate psudotime
    print('end paga')
    
    ## Calculate euclidean distance
    print('start Euclidean')
    coordiantes = adata.obs.sort_values(by="dpt_pseudotime")[["n_col","n_row"]]
    dist =  pd.DataFrame(euclidean_distances(coordiantes,coordiantes))
    dist = Norm(dist[0])
    dist.index, dist.name = coordiantes.index, "Distance"
    adata.obs = adata.obs.join(dist)
    print('end Euclidean')
    
    return adata


def main(args):
    if args.feature_path is not None:
        if os.path.isdir(args.feature_path):
            feature_files = [_ for _ in os.listdir(args.feature_path) if not _.startswith('.') and _.endswith('.csv')]
        else:
            feature_files = [args.feature_path]
    

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print(feature_files)
    for f in feature_files:
        wsi_df = pd.read_csv(os.path.join(args.feature_path,f))
        adata =  paga(wsi_df)
        adata.write(os.path.join(args.output_path,"{}.h5ad".format(f)), compression="gzip")
  
     
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Feature Extraction by phikon')

    parser.add_argument('--feature_path', default=os.path.join(os.getcwd(), 'results/'), type=str,
                        help="path for extract image features in csv format")
    parser.add_argument('--output_path', default=os.path.join(os.getcwd(), 'output/'), type=str,
                        help='the relative path to output the result')

    args = parser.parse_args()

    main(args)
