import numpy as np
import utils

# RNG
rng = np.random.default_rng(42)

# Square block matrix

# Matrix block shape
M = 3
N = 4

# Block shape
m = 2
n = 2

# Unique block generation
num_unq_blocks = M + N - 1
tpltz_blks = []
hnkl_blks = []

for i in range(num_unq_blocks):
    tpltz_blks.append(utils.random_toeplitz(m, n, rng, low_rank=True))
    hnkl_blks.append(utils.random_hankel(m, n, rng, low_rank=True))

# Total block matrix generation for SVD comparison
block_hankel_matrix = utils.blocks_to_hankel(M, N, hnkl_blks, (m, n))
block_toeplitz_matrix = utils.blocks_to_toeplitz(M, N, tpltz_blks, (m, n))

# Full SVD computation
U_hankel, E_hankel, Vh_hankel = utils.method_of_snapshots(
    block_hankel_matrix, full_matrices=False, truncate_tol=0.6
)
print(
    "Method of snapshots:",
    np.linalg.norm(block_hankel_matrix - U_hankel @ np.diag(E_hankel) @ Vh_hankel),
)
U_toeplitz, E_toeplitz, Vh_toeplitz = np.linalg.svd(
    block_toeplitz_matrix, full_matrices=False
)

U_hankel, E_hankel, Vh_hankel = utils.svd_with_tol(
    block_hankel_matrix, full_matrices=False, truncate_tol=0.6
)
print(
    "Standard SVD:",
    np.linalg.norm(block_hankel_matrix - U_hankel @ np.diag(E_hankel) @ Vh_hankel),
)
U_toeplitz, E_toeplitz, Vh_toeplitz = np.linalg.svd(
    block_toeplitz_matrix, full_matrices=False
)
