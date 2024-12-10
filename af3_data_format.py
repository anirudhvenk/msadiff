import torch
import json
import os
import re
import pandas as pd
import shutil

from data import read_msa
from tqdm import tqdm
from unipressed import UniprotkbClient

base_dir = "databases/openfold/filtered"
fm_output_dir = "flow_matching/data/openfold_filtered_data"
protenix_output_dir = "flow_matching/protenix/openfold_filtered"
output_json_path = "flow_matching/protenix/openfold_filtered_input.json"

train_list = pd.read_csv("databases/openfold/train_msas.csv", header=None)[0].tolist()
train_set = set(train_list)

with open(output_json_path, 'w') as json_file:
    json_file.write('[\n')
    first_entry = True

    for seq_id in tqdm(os.listdir(base_dir)):
        if seq_id in train_set:
            msa_path = os.path.join(base_dir, seq_id, "a3m/uniclust30.a3m")
            msa = read_msa(msa_path)

            fm_dir = os.path.join(fm_output_dir, seq_id)
            os.makedirs(fm_dir, exist_ok=True)
            shutil.copy(msa_path, os.path.join(fm_dir, "non_pairing.a3m"))

            protenix_dir = os.path.join(protenix_output_dir, seq_id, 'msa', '1')
            os.makedirs(protenix_dir, exist_ok=True)
            shutil.copy(msa_path, os.path.join(protenix_dir, 'non_pairing.a3m'))

            af3_json = {
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": msa[0][1],
                            "count": 1,
                            "msa": {
                                "precomputed_msa_dir": f"openfold_filtered/{seq_id}/msa/1",
                                "pairing_db": "uniref100"
                            }
                        }   
                    }
                ],
                "modelSeeds": [],
                "assembly_id": "1",
                "name": seq_id
            }

            if not first_entry:
                json_file.write(',\n')
            json.dump(af3_json, json_file, indent=4)
            first_entry = False

    json_file.write('\n]')