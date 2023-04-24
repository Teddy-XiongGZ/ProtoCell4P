from scipy.io import mmread
from scipy import sparse
import time

start = time.time()

dat = mmread("DCM_HCM_Expression_Matrix_raw_counts_V1.mtx")  # this step is very slow

sparse.save_npz("raw_counts.npz", sparse.csr_matrix(dat.transpose()))

print("Time spent: {}s".format(time.time() - start))