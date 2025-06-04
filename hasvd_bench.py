# %%
import utils
import numpy as np
import time


def nodemap_hasvd(tree: utils.hasvd_Node, A_array, m, n, M, N):
    """
    Create a mapping of nodes to their corresponding matrix blocks in a distributed HASVD tree.
    Parameters
    ----------
    tree : utils.hasvd_Node
        The root node of the tree.
    Returns
    -------
    node_to_block : dict
        A dictionary mapping nodes to their corresponding matrix blocks.
    """

    node_to_block = {}
    for node in tree.traverse():

        # Check if the node is a leaf node
        if node.is_leaf:
            block_coord = (
                np.floor_divide(node.tag - 1, N) * m,
                ((node.tag - 1) % N) * n,
            )
            block_shape = (m, n)
            # Get the block corresponding to the node
            block = utils.array_to_hankel(
                A_array,
                (m * M, n * N),
                block_coord,
                block_shape,
            )
            node_to_block[node] = block
    return node_to_block


def naive_error(
    node: utils.hasvd_Node,
    eps_star: float,
    omega: float,
    num_nonleaf_nodes: int,
):
    """
    Calculate the naive error for a node in the HASVD tree.

    Parameters
    ----------
    node : utils.hasvd_Node
        The node for which to calculate the naive error.
    eps_star : float
        The epsilon star value.
    omega : float
        The omega value.
    num_nonleaf_nodes : int
        The number of non-leaf nodes in the tree.

    Returns
    -------
    error : float
        The naive error for the node.
    """
    if node.is_leaf:
        num_neighbor_nodes = 0
        sum_shapes = 0
        for child in node.parent.children:
            if child.is_leaf:
                num_neighbor_nodes += 1
                sum_shapes += child.m * child.n
        return (
            (1 - omega)
            * eps_star
            / np.sqrt(num_neighbor_nodes)
            * ((sum_shapes) / (node.parent.m * node.parent.n))
            * (1 / num_nonleaf_nodes)
        )
    elif node.is_root:
        return omega * eps_star
    else:
        return (
            (1 - omega)
            * eps_star
            * ((node.m * node.n) / (node.parent.m * node.parent.n))
            * (1 / num_nonleaf_nodes)
        )


def tight_error(
    node: utils.hasvd_Node,
    eps_star: float,
    omega: float,
    num_nonleaf_nodes: int,
):
    """
    Calculate the naive error for a node in the HASVD tree.

    Parameters
    ----------
    node : utils.hasvd_Node
        The node for which to calculate the naive error.
    eps_star : float
        The epsilon star value.
    omega : float
        The omega value.
    num_nonleaf_nodes : int
        The number of non-leaf nodes in the tree.

    Returns
    -------
    error : float
        The naive error for the node.
    """
    if node.is_leaf:
        return (
            (1 - omega)
            * eps_star
            * np.sqrt((node.m * node.n) / (node.root.m * node.root.n))
            * (1 / (num_nonleaf_nodes + 1))
        )
    elif node.is_root:
        return omega * eps_star
    else:
        sum_shapes = 0
        for child in node.parent.children:
            if (not child.is_leaf) and (child != node):
                sum_shapes += child.m * child.n
        return (
            (1 - omega)
            * eps_star
            * ((node.m * node.n) / (sum_shapes))
            * (1 / (num_nonleaf_nodes + 1))
        )


def bench(M, N, m, n, eps, omega):
    M = M
    N = N
    m = m
    n = n
    rng = np.random.default_rng(12)
    eps = eps
    omega = omega
    # Create a random matrix with block size m x n with M xN blocks
    rng = np.random.default_rng(12)
    A = utils.random_hankel(m * M, n * N, rng)
    # print("A:", A)
    rank = np.linalg.matrix_rank(A)

    A_array = utils.hankel_to_array(A)

    start_time = time.time()
    # SVD of the whole matrix
    U, E, Vh = utils.svd_with_tol(
        A,
        full_matrices=False,
        truncate_tol=eps,
    )
    runtime_svd = time.time() - start_time
    # print("SVD of the whole matrix")
    err_svd = np.linalg.norm(A - U @ np.diag(E) @ Vh)
    rk = len(E)
    # print("Error:", err_svd)

    # DISTRIBUTED HASVD
    # print("\u0332".join("Distributed HASVD"))
    # Create a tree for the distributed HASVD
    tree = utils.two_level_bidir_dist_hasvd_tree(M, N, 1, 0, block_shape=(m, n))
    # utils.draw_nxgraph(tree)

    # Create mapping of nodes to their corresponding matrix blocks
    node_to_block = nodemap_hasvd(
        tree,
        A_array,
        m,
        n,
        M,
        N,
    )

    # Create a function to get the block for a node
    def get_dist_hasvd_block(node):
        return node_to_block[node]

    # Create mapping of nodes to their corresponding naive error

    # Sum non-leaf nodes
    num_nonleaf_nodes = 0
    for node in tree.traverse():
        if not node.is_leaf:
            num_nonleaf_nodes += 1

    # NAIVE ERROR

    # Create a function to get the naive error for a node
    def get_dist_naive_error(node):
        return naive_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start_time = time.time()
    U1, E1, Vh1 = utils.hasvd(
        tree, get_dist_hasvd_block, local_eps=get_dist_naive_error
    )
    runtime_dist_naive = time.time() - start_time
    err_dist_naive = np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1)
    rk_dist_naive = len(E1)
    # print("Naive Error:", err_dist_naive)

    # TIGHT ERROR

    # Create a function to get the tight error for a node
    def get_dist_tight_error(node):
        return tight_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start_time = time.time()
    U1, E1, Vh1 = utils.hasvd(
        tree, get_dist_hasvd_block, local_eps=get_dist_tight_error
    )
    runtime_dist_tight = time.time() - start_time
    err_dist_tight = np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1)
    rk_dist_tight = len(E1)
    # print("Tight Error:", err_dist_tight)

    # INCREMENTAL HASVD
    # print("\u0332".join("Incremental HASVD"))
    # Create a tree for the incremental HASVD
    tree = utils.two_level_bidir_inc_hasvd_tree(M, N, 1, 0, (m, n))
    # utils.draw_nxgraph(tree)
    # Create mapping of nodes to their corresponding matrix blocks
    node_to_block = nodemap_hasvd(
        tree,
        A_array,
        m,
        n,
        M,
        N,
    )

    # Create a function to get the block for a node
    def get_inc_hasvd_block(node):
        return node_to_block[node]

    # Sum non-leaf nodes
    num_nonleaf_nodes = 0
    for node in tree.traverse():
        if not node.is_leaf:
            num_nonleaf_nodes += 1

    # NAIVE ERROR

    # Create a function to get the naive error for a node
    def get_inc_naive_error(node):
        return naive_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start_time = time.time()
    U2, E2, Vh2 = utils.hasvd(tree, get_inc_hasvd_block, local_eps=get_inc_naive_error)
    runtime_inc_naive = time.time() - start_time
    err_inc_naive = np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2)
    rk_inc_naive = len(E2)
    # print("Error:", err_inc_naive)

    # TIGHT ERROR
    # Create a function to get the tight error for a node
    def get_inc_tight_error(node):
        return tight_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start_time = time.time()
    U2, E2, Vh2 = utils.hasvd(tree, get_inc_hasvd_block, local_eps=get_inc_tight_error)
    runtime_inc_tight = time.time() - start_time
    err_inc_tight = np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2)
    rk_inc_tight = len(E2)
    # print("Error:", err_inc_tight)

    return (
        A.shape,
        rank,
        err_svd,
        rk,
        err_dist_naive,
        rk_dist_naive,
        err_dist_tight,
        rk_dist_tight,
        err_inc_naive,
        rk_inc_naive,
        err_inc_tight,
        rk_inc_tight,
    )


if __name__ == "__main__":
    # All combinations of arguments for the benchmark
    M = [100]
    N = [200]
    m = [10]
    n = [20]
    eps = [
        0.000001,
        0.000005,
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
    ]
    omega = [0.1]

    # Create a list of all combinations of arguments
    combinations = [
        (M_val, N_val, m_val, n_val, eps_val, omega_val)
        for M_val in M
        for N_val in N
        for m_val in m
        for n_val in n
        for eps_val in eps
        for omega_val in omega
    ]
    # Prepare dataframe for inputs
    import pandas as pd

    df = pd.DataFrame(
        combinations,
        columns=[
            "M",
            "N",
            "m",
            "n",
            "eps",
            "omega",
        ],
    )
    # Add columns for results, initialized with NaN
    df["rank"] = np.nan
    df["error"] = np.nan
    df["rk"] = np.nan
    df["error_dist_naive"] = np.nan
    df["rk_dist_naive"] = np.nan
    df["error_dist_tight"] = np.nan
    df["rk_dist_tight"] = np.nan
    df["error_inc_naive"] = np.nan
    df["rk_inc_naive"] = np.nan
    df["error_inc_tight"] = np.nan
    df["rk_inc_tight"] = np.nan

    # Run the benchmark for each combination of arguments and immediately store the results in csv after loop
    for i, (M_val, N_val, m_val, n_val, eps_val, omega_val) in enumerate(combinations):
        print(f"Running benchmark {i + 1}/{len(combinations)}")
        (
            shape,
            rank,
            err_svd,
            rk,
            err_dist_naive,
            rk_dist_naive,
            err_dist_tight,
            rk_dist_tight,
            err_inc_naive,
            rk_inc_naive,
            err_inc_tight,
            rk_inc_tight,
        ) = bench(M_val, N_val, m_val, n_val, eps_val, omega_val)
        df.loc[i] = [
            M_val,
            N_val,
            m_val,
            n_val,
            eps_val,
            omega_val,
            rank,
            err_svd,
            rk,
            err_dist_naive,
            rk_dist_naive,
            err_dist_tight,
            rk_dist_tight,
            err_inc_naive,
            rk_inc_naive,
            err_inc_tight,
            rk_inc_tight,
        ]

        # Save the dataframe to a CSV file
        df.to_csv("benchmark_results_2.csv", index=False)


# %%
