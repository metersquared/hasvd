# %%
import utils
import numpy as np


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


if __name__ == "__main__":

    M = 4
    N = 3
    m = 5
    n = 2
    rng = np.random.default_rng(12)
    eps = 1e-3
    omega = 0.1

    # Create a random matrix with block size m x n with M xN blocks
    rng = np.random.default_rng(12)
    A = utils.random_hankel(m * M, n * N, rng)
    # print("A:", A)
    print("Matrix rank:", np.linalg.matrix_rank(A))
    print("Matrix shape:", A.shape)

    A_array = utils.hankel_to_array(A)
    print("A_array:", A_array)

    # SVD of the whole matrix
    U, E, Vh = utils.svd_with_tol(
        A,
        full_matrices=False,
        truncate_tol=eps,
    )
    print("SVD of the whole matrix")
    print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))

    # DISTRIBUTED HASVD
    print("\u0332".join("Distributed HASVD"))
    # Create a tree for the distributed HASVD
    tree = utils.two_level_bidir_dist_hasvd_tree(M, N, 1, 0, block_shape=(m, n))
    utils.draw_nxgraph(tree)

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
    U1, E1, Vh1 = utils.hasvd(
        tree, get_dist_hasvd_block, local_eps=get_dist_naive_error
    )
    print("Naive Error:", np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1))

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
    U1, E1, Vh1 = utils.hasvd(
        tree, get_dist_hasvd_block, local_eps=get_dist_tight_error
    )
    print("Tight Error:", np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1))

    # INCREMENTAL HASVD
    print("\u0332".join("Incremental HASVD"))
    # Create a tree for the incremental HASVD
    tree = utils.two_level_bidir_inc_hasvd_tree(M, N, 1, 0, (m, n))
    utils.draw_nxgraph(tree)
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
    U2, E2, Vh2 = utils.hasvd(tree, get_inc_hasvd_block, local_eps=get_inc_naive_error)
    print("Error:", np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2))

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
    U2, E2, Vh2 = utils.hasvd(tree, get_inc_hasvd_block, local_eps=get_inc_tight_error)
    print("Error:", np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2))
# %%
