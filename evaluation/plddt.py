import os
import esm
import torch
import biotite.structure.io as bsio
from typing import List, Dict, Union
from tqdm import tqdm
import numpy as np
import torch.distributed as dist


class ESMMetric:
    def __init__(self, device: str = "cpu"):
        self.model = esm.pretrained.esmfold_v1().eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, protein: str) -> float:
        try:
            output = self.model.infer_pdb(protein)

            tmp_path = "./tmp"
            os.makedirs(tmp_path, exist_ok=True)
            file_path = os.path.join(tmp_path, f"result-{np.random.randint(low=0, high=1000)}-{self.device}.pdb")
            with open(file_path, "w") as f:
                f.write(output)
            struct = bsio.load_structure(file_path, extra_fields=["b_factor"])
            return struct.b_factor.mean()
        except Exception:
            print(protein)
            return 0.


def compute_plddt(proteins: List[str], device="cuda") -> List[Dict[str, Union[str, float]]]:
    metric_fn = ESMMetric(device)

    result = []
    for protein in tqdm(proteins):
        result.append(
            {
                "protein": protein,
                "pLDDT": metric_fn(protein=protein)
            }
        )
    return result
