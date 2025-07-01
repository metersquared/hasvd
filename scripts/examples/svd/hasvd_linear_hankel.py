# %%
import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
import numpy as np
import time

# %% [markdown]
# We generate a random matrix with $M\times N$ blocks with each block being $m\times n$

# %%
rng = np.random.default_rng(42)

# Number of blocks
partition = 50
# Block size
m = 1000
n = 50
# Direction of partitioning
direction = 0
# Rank of matrix

if direction == 0:
    M = 1
    N = partition
else:
    M = partition
    N = 1

A = matrix.random_hankel(M * m, N * n, rng=rng)
rank = np.linalg.matrix_rank(A)
condition_num = np.linalg.cond(A)

print(
    "Block number:",
    M,
    "x",
    N,
    ", Block shape",
    m,
    "x",
    n,
    ", Shape",
    M * m,
    "x",
    N * n,
    ", rank:",
    rank,
    ", condition num:",
    condition_num,
)

# %%
# APPROX. WITH LAPACK SVD
print("\u0332".join("Approx. with LAPACK SVD"))

# Tolerance for SVD
tol = 1e-12
start = time.time()
U, S, Vt = svd.svd_with_tol(A, truncate_tol=tol)
duration = time.time() - start
print("Time for SVD:", duration, "seconds")
print("Rank of approximation:", len(S))
print("Frobenius norm of error:", np.linalg.norm(A - U @ np.diag(S) @ Vt))

# Array representation
A_array = matrix.hankel_to_array(A)

# %%
# APPROX. WITH BLOCK HASVD
print("\u0332".join("Approx. with Linear Distributed HASVD"))

# Control parameter
omega = 0.01

# Tree construction
tree = trees.dist_hasvd_tree(partition, direction, (m, n))

"""
def node_to_block_map(node: trees.hasvd_Node):
    if direction == 0:
        return A[:, node.tag * n : (node.tag + 1) * n]
    else:
        return A[node.tag * m : (node.tag + 1) * m, :]


non_leaf_count = trees.non_leaf_count(tree)
branch_count = trees.branch_node_count(tree)

# Naive error in HASVD
print("Naive error in HASVD:")


def nodal_error(node):
    return errors.naive_error(node, tol, omega, non_leaf_count)


# Compute HASVD approximation
start = time.time()
U, S, Vt, ranks = svd.hasvd(
    tree, node_to_block_map, local_eps=nodal_error, track_ranks=True
)
duration = time.time() - start
print("Time for SVD:", duration, "seconds")

print("Rank of approximation:", len(S))
print("Frobenius norm of error:", np.linalg.norm(A - U @ np.diag(S) @ Vt))
svd.rank_analysis(tree, ranks)

# Tight error in HASVD
print("Tight error in HASVD:")


def nodal_error(node):
    return errors.tight_error(node, tol, omega, branch_count)


# Compute HASVD approximation

U, S, Vt, ranks = svd.hasvd(
    tree, node_to_block_map, local_eps=nodal_error, track_ranks=True
)

print("Rank of approximation:", len(S))
print("Frobenius norm of error:", np.linalg.norm(A - U @ np.diag(S) @ Vt))
svd.rank_analysis(tree, ranks)

# %%
# APPROX. WITH BLOCK INC HASVD
print("\u0332".join("Approx. with Linear Incremental HASVD"))


# Tree construction
tree = trees.inc_hasvd_tree(partition, direction, (m, n))


def node_to_block_map(node: trees.hasvd_Node):
    if direction == 0:
        return A[:, node.tag * n : (node.tag + 1) * n]
    else:
        return A[node.tag * m : (node.tag + 1) * m, :]


non_leaf_count = trees.non_leaf_count(tree)
branch_count = trees.branch_node_count(tree)

# Naive error in HASVD
print("Naive error in HASVD:")


def nodal_error(node):
    return errors.naive_error(node, tol, omega, non_leaf_count)


# Compute HASVD approximation
start = time.time()
U, S, Vt, ranks = svd.hasvd(
    tree, node_to_block_map, local_eps=nodal_error, track_ranks=True
)
duration = time.time() - start
print("Time for SVD:", duration, "seconds")

print("Rank of approximation:", len(S))
print("Frobenius norm of error:", np.linalg.norm(A - U @ np.diag(S) @ Vt))
svd.rank_analysis(tree, ranks)

# Tight error in HASVD
print("Tight error in HASVD:")


def nodal_error(node):
    return errors.tight_error(node, tol, omega, branch_count)


# Compute HASVD approximation
start = time.time()
U, S, Vt, ranks = svd.hasvd(
    tree, node_to_block_map, local_eps=nodal_error, track_ranks=True
)
duration = time.time() - start
print("Time for SVD:", duration, "seconds")

print("Rank of approximation:", len(S))
print("Frobenius norm of error:", np.linalg.norm(A - U @ np.diag(S) @ Vt))
svd.rank_analysis(tree, ranks)


# Uniform error in HASVD
print("Uniform error in HASVD:")


def nodal_error(node):
    return tol


# Compute HASVD approximation
start = time.time()
U, S, Vt, ranks = svd.hasvd(
    tree, node_to_block_map, local_eps=nodal_error, track_ranks=True
)
duration = time.time() - start
print("Time for SVD:", duration, "seconds")

print("Rank of approximation:", len(S))
print("Frobenius norm of error:", np.linalg.norm(A - U @ np.diag(S) @ Vt))
svd.rank_analysis(tree, ranks)
"""
