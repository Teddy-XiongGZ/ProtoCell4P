import pandas as pd
from scipy import sparse
import time

start = time.time()

dat = pd.read_csv("20210220_NasalSwab_RawCounts.txt", sep='\t')  # this step is very slow

with open("barcodes.txt", 'w') as f:
    f.write('\n'.join(dat.columns))
with open("genes.txt", 'w') as f:
    f.write('\n'.join(dat.index))
sparse.save_npz("RawCounts.npz", sparse.csr_matrix(dat.transpose()))

print("Time spent: {}s".format(time.time() - start))
