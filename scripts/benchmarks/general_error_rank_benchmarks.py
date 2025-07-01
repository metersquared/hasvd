# %%
import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
from functools import partial
import numpy as np


M = 20
N = 20
m = 100
n = 100
rng = np.random.Generator(np.random.MT19937(42))
trials = 3
direction = 1
set_rank = 20

if direction == 0:
    partition = N
    block_n = n
    block_m = M * m
else:
    partition = M
    block_n = N * n
    block_m = m

tree_list = [
    trees.dist_hasvd_tree(partition, direction, (block_m, block_n)),
    trees.inc_hasvd_tree(partition, direction, (block_m, block_n)),
]


def bench(tol, omega):

    err_lapack = [None] * trials
    rank = [None] * trials

    err_naive = [None] * len(tree_list)
    err_tight = [None] * len(tree_list)
    rank_naive = [None] * len(tree_list)
    rank_tight = [None] * len(tree_list)

    for i in range(len(tree_list)):
        err_naive[i] = [None] * trials
        err_tight[i] = [None] * trials
        rank_naive[i] = [None] * trials
        rank_tight[i] = [None] * trials

    for i in range(trials):
        A = matrix.random_matrix(m * M, n * N, set_rank, rng, 1e3)
        U, S, Vt = svd.svd_with_tol(A, truncate_tol=tol)
        err_lapack[i] = np.linalg.norm(A - U @ np.diag(S) @ Vt)
        rank[i] = len(S)

        def node_to_block_map(node: trees.hasvd_Node):
            if direction == 0:
                return A[:, node.tag * n : (node.tag + 1) * n]
            else:
                return A[node.tag * m : (node.tag + 1) * m, :]

        for idx, tree in enumerate(tree_list):

            non_leaf_count = trees.non_leaf_count(tree)
            branch_count = trees.branch_node_count(tree)
            """
            def nodal_error(node):
                return errors.naive_error(node, tol, omega, non_leaf_count)

            U, S, Vt = svd.hasvd(tree, node_to_block_map, local_eps=nodal_error)
            err_naive[idx][i] = np.linalg.norm(A - U @ np.diag(S) @ Vt)
            rank_naive[idx][i] = len(S)
            """

            def nodal_error(node):
                return errors.tight_error(node, tol, omega, branch_count)

            U, S, Vt = svd.hasvd(tree, node_to_block_map, local_eps=nodal_error)
            err_tight[idx][i] = np.linalg.norm(A - U @ np.diag(S) @ Vt)
            rank_tight[idx][i] = len(S)

    err_lapack = np.mean(err_lapack)
    rank = np.mean(rank)
    for i in range(len(tree_list)):
        # err_naive[i] = np.mean(err_naive[i])
        err_tight[i] = np.mean(err_tight[i])
        # rank_naive[i] = np.mean(rank_naive[i])
        rank_tight[i] = np.mean(rank_tight[i])

    return err_lapack, rank, err_tight, rank_tight


if __name__ == "__main__":
    eps = [4e-2, 2e-2, 9e-3, 7e-3, 5e-3, 3e-3, 1e-3, 8e-4, 6e-4, 4e-4]
    omegas = [0.1, 0.25, 0.5, 0.75, 0.9]

    combinations = [(eps_val, omega_val) for eps_val in eps for omega_val in omegas]

    import pandas as pd

    df = pd.DataFrame(
        combinations,
        columns=[
            "eps",
            "omega",
        ],
    )

    df["err"] = np.nan
    df["rk"] = np.nan
    # df["err_naive_0"] = np.nan
    # df["rk_naive_0"] = np.nan
    df["err_tight_0"] = np.nan
    df["rk_tight_0"] = np.nan

    # df["err_naive_1"] = np.nan
    # df["rk_naive_1"] = np.nan
    df["err_tight_1"] = np.nan
    df["rk_tight_1"] = np.nan

    for i, (eps_val, omega_val) in enumerate(combinations):
        print(f"Running benchmark {i + 1}/{len(combinations)}")
        (err_lapack, rank, err_tight, rank_tight) = bench(eps_val, omega_val)
        df.loc[i] = [
            eps_val,
            omega_val,
            err_lapack,
            rank,
            # err_naive[0],
            # rank_naive[0],
            err_tight[0],
            rank_tight[0],
            # err_naive[1],
            # rank_naive[1],
            err_tight[1],
            rank_tight[1],
        ]

        df.to_csv("general_error_rank_data_bench_2.csv", index=False)
