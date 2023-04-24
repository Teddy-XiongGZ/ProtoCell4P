import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import tqdm
from torch.utils.data import Dataset

class OurDataset(Dataset):
    def __init__(self, X, y, cell_id=None, gene_id=None, class_id=None, ct=None, ct_id=None):
        self.X = X
        self.y = y
        self.cell_id = cell_id
        self.gene_id = gene_id
        self.class_id = class_id
        self.ct = ct
        self.ct_id = ct_id
    def __getitem__(self, i):
        if self.ct_id is not None:
            return self.X[i], self.y[i], self.ct[i]
        return self.X[i], self.y[i]
    def __len__(self):
        return len(self.y)

def load_lupus(data_path = "../data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad", task=None, load_ct=True, keep_sparse=True):
    # https://github.com/yelabucsf/lupus_1M_cells_clean
    assert task is not None
    
    adata = sc.read_h5ad(data_path)
    
    # before: (834096, 32738) | after: (834096, 24205)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.scale(adata, max_value=10, zero_center=True) # Unable to allocate 75.2 GiB

    if keep_sparse is False:
        adata.X = adata.X.toarray()

    print("Preprocessing Complete!")
    
    X = []
    y = []

    genes = adata.var_names.tolist()
    barcodes = adata.obs_names.tolist()
    cell_types = adata.obs["ct_cov"]

    ct_id = sorted(set(cell_types))
    mapping_ct = {c:idx for idx, c in enumerate(ct_id)}    
    ct = []

    for ind in tqdm.tqdm(sorted(set(adata.obs["ind_cov"]))):
        disease = list(set(adata.obs[adata.obs["ind_cov"] == ind]["disease_cov"]))
        pop = list(set(adata.obs[adata.obs["ind_cov"] == ind]["pop_cov"]))
        assert len(disease) == 1
        assert len(pop) == 1
        x = adata.X[adata.obs["ind_cov"] == ind]
        X.append(x)
        if task.lower() == "disease":
            y.append(disease[0])
        elif task.lower() == "population" or task.lower() == "pop":
            y.append(pop[0])
        ct.append([mapping_ct[c] for c in cell_types[adata.obs["ind_cov"] == ind]])

    class_id = sorted(set(y))
    mapping = {c:idx for idx, c in enumerate(class_id)}
    y = [mapping[c] for c in y]
    
    print(("[Size of dataset] "+" | ".join(["{:s}: {:d}"] * len(class_id))).format(*[item for i in range(len(class_id)) for item in [class_id[i], y.count(i)]]))

    # each sample in X has 4935 cells in average
    if load_ct:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
    else:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)

def load_cardio(data_dir = "../data/cardio", load_ct=True, keep_sparse=True):
    dat = sparse.load_npz(os.path.join(data_dir, "raw_counts.npz"))
    # genes = open(os.path.join(data_dir, "SCP1303expression614a0209771a5b0d7f033712DCM_HCM_Expression_Matrix_genes_V1.tsv")).read().strip().split("\n")
    genes = pd.read_csv(os.path.join(data_dir, "DCM_HCM_Expression_Matrix_genes_V1.tsv"), sep="\t", header=None).iloc[:,1].tolist()
    barcodes = open(os.path.join(data_dir, "DCM_HCM_Expression_Matrix_barcodes_V1.tsv")).read().strip().split("\n")
    meta = pd.read_csv(os.path.join(data_dir, "DCM_HCM_MetaData_V1.txt"), sep="\t").drop(axis=0,index=0).reset_index(drop=True)

    assert dat.shape[0] == len(barcodes) and len(barcodes) == meta.shape[0]
    assert dat.shape[1] == len(genes)
    
    cell_types = meta.cell_type__ontology_label
    ct_id = sorted(set(cell_types))
    mapping_ct = {c:idx for idx, c in enumerate(ct_id)}

    X = []
    y = []
    ct = []

    adata = sc.AnnData(dat.astype(np.float32), obs=barcodes, var=genes)

    # before: (592689, 36601) | after: (592689, 32151)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.scale(adata, max_value=10, zero_center=True)

    barcodes = adata.obs[0].tolist()
    genes = adata.var[0].tolist()
    
    if keep_sparse is False:
        adata.X = adata.X.toarray()

    for ind in tqdm.tqdm(sorted(set(meta.donor_id))):
        disease = list(set(meta.disease__ontology_label[meta.donor_id == ind]))
        assert len(disease) == 1
        x = adata.X[meta.donor_id == ind]
        X.append(x)
        y.append(disease[0])
        ct.append([mapping_ct[c] for c in cell_types[meta.donor_id == ind]])
    
    class_id = sorted(set(y))
    mapping = {c:idx for idx, c in enumerate(class_id)}
    y = [mapping[c] for c in y]
    
    # [Size of dataset] dilated cardiomyopathy: 11 | hypertrophic cardiomyopathy: 15 | normal: 16
    print(("[Size of dataset] "+" | ".join(["{:s}: {:d}"] * len(class_id))).format(*[item for i in range(len(class_id)) for item in [class_id[i], y.count(i)]]))
    
    # each sample in X has 14111 cells in average
    if load_ct:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
    else:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)    


def load_covid(data_dir = "../data/covid", load_ct=True, keep_sparse=True):
    dat = sparse.load_npz(os.path.join(data_dir, "RawCounts.npz"))
    genes = open(os.path.join(data_dir, "genes.txt")).read().strip().split("\n")
    barcodes = open(os.path.join(data_dir, "barcodes.txt")).read().strip().split("\n")
    meta = pd.read_csv(os.path.join(data_dir, "20210701_NasalSwab_MetaData.txt"), sep="\t").drop(axis=0,index=0).reset_index(drop=True)

    cell_types = pd.read_csv(os.path.join(data_dir, "20210220_NasalSwab_UMAP.txt"), sep="\t").drop(axis=0,index=0).reset_index(drop=True)["Category"]
    ct_id = sorted(set(cell_types))
    mapping_ct = {c:idx for idx, c in enumerate(ct_id)}

    X = []
    y = []
    ct = []

    adata = sc.AnnData(dat.astype(np.float32), obs=barcodes, var=genes)
    # adata = sc.AnnData(dat.astype(np.float32))
    # before: (32588, 32871) | after: (32588, 29696)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.scale(adata, max_value=10, zero_center=True)

    barcodes = adata.obs[0].tolist()
    genes = adata.var[0].tolist()
    
    if keep_sparse is False:
        adata.X = adata.X.toarray()

    for ind in tqdm.tqdm(sorted(set(meta.donor_id))):
        disease = list(set(meta.disease__ontology_label[meta.donor_id == ind]))
        assert len(disease) == 1
        if disease[0] == "long COVID-19" or disease[0] == "respiratory failure":
            continue
        x = adata.X[meta.donor_id == ind]
        X.append(x)
        y.append(disease[0])
        ct.append([mapping_ct[c] for c in cell_types[meta.donor_id == ind]])
    
    class_id = sorted(set(y))
    mapping = {c:idx for idx, c in enumerate(class_id)}
    y = [mapping[c] for c in y]
    
    # [Size of dataset] COVID-19: 35 | long COVID-19: 2 | normal: 15 | respiratory failure: 6
    print(("[Size of dataset] "+" | ".join(["{:s}: {:d}"] * len(class_id))).format(*[item for i in range(len(class_id)) for item in [class_id[i], y.count(i)]]))
    
    # each sample in X has 562 cells in average
    if load_ct:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
    else:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)    
