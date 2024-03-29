import argparse
import subprocess
import os
import re
import pandas as pd

def save_cluster_sizes(df, exp_name, output_csv, mmseq_params):
    clusters = df[0].to_list()
    out_path = output_csv.split('.csv')[0] + f'{mmseq_params}_cluster_sizes.csv'
    if os.path.exists(out_path):
        res_df = pd.read_csv(out_path)
    else:
        res_df = pd.DataFrame({})
    
    res_df.loc[:, exp_name] = clusters
    res_df.to_csv(out_path, index = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fasta_dir")
    parser.add_argument("output_csv")
    parser.add_argument("exp_name")
    args = parser.parse_args()

    seq_identity = 0.9
    c = 0.8
    cov_mode = 0
    to_resign = '--cluster-reassign' #'--cluster-reassign'
    mmseq_params = f'id{seq_identity}_c{c}_cov-mode{cov_mode}_{to_resign}' 

    # create mmseq
    subprocess.call(
        f"mmseqs easy-cluster {args.input_fasta_dir} set tmp --min-seq-id {seq_identity} -c {c} --cov-mode {cov_mode} {to_resign}",
        shell=True, stdout=subprocess.DEVNULL)

    df = pd.read_table('set_cluster.tsv', header=None)
    
    density = df[0].unique().shape[0]/2048
    save_cluster_sizes(df, args.exp_name, args.output_csv, mmseq_params)

    if os.path.exists(args.output_csv):
        res_df = pd.read_csv(args.output_csv)
    else:
        res_df = pd.DataFrame({"file": [], "density_" + mmseq_params: []})
    
    res_df = res_df.set_index('file')
    new_row = pd.DataFrame({"file": [args.exp_name], "density_" + mmseq_params: [density]})
    # res_df = res_df.set_index('file')
    # res_df = pd.concat([new_row, res_df])
    res_df.loc[args.exp_name, "density_" + mmseq_params] = density

    res_df = res_df.reset_index()

    res_df.to_csv(args.output_csv, index = False)
#     print(f'''
#     For {args.input_fasta_dir} the cluster density is {density} 
# ''')

