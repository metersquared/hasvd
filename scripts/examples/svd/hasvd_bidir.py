# %%
import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
import time
import numpy as np


def nodemap_hasvd(tree: trees.hasvd_Node, A_array, m, n, M, N):
    """
    Create a mapping of nodes to their corresponding matrix blocks in a internal linear HASVD tree.
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
                np.floor_divide(node.tag, N) * m,
                ((node.tag) % N) * n,
            )
            block_shape = (m, n)
            # Get the block corresponding to the node
            block = matrix.array_to_hankel(
                A_array,
                (m * M, n * N),
                block_coord,
                block_shape,
            )
            node_to_block[node] = block
    return node_to_block


if __name__ == "__main__":

    M = 10
    N = 10
    m = 1000
    n = 1000
    rng = np.random.default_rng(12)
    eps = 1.0e-3
    omega = 0.1

    # Create a random matrix with block size m x n with M xN blocks
    rng = np.random.default_rng(12)
    A = matrix.random_hankel(m * M, n * N, rng)
    # print("A:", A)
    print("Matrix rank:", np.linalg.matrix_rank(A))
    print("Matrix shape:", A.shape)

    A_array = matrix.hankel_to_array(A)
    print("A_array:", A_array)

    # SVD of the whole matrix
    start = time.time()
    U, E, Vh = svd.svd_with_tol(
        A,
        full_matrices=False,
        truncate_tol=eps,
    )
    duration = time.time() - start
    print("SVD of the whole matrix")
    print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))
    print("Time of SVD:", duration)
    # DISTRIBUTED HASVD
    print("\u0332".join("Distributed HASVD"))
    # Create a tree for the distributed HASVD
    tree = trees.two_level_bidir_dist_hasvd_tree(M, N, 1, 0, block_shape=(m, n))

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

    # Sum of branching nodes
    num_branching_nodes = 0
    for node in tree.traverse():
        if not node.is_leaf and any(not child.is_leaf for child in node.children):
            num_branching_nodes += 1

    # NAIVE ERROR

    # Create a function to get the naive error for a node
    def get_dist_naive_error(node):
        return errors.naive_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U1, E1, Vh1 = svd.hasvd(tree, get_dist_hasvd_block, local_eps=get_dist_naive_error)
    duration = time.time() - start
    print("Naive Error:", np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1))
    print("Time of SVD:", duration)

    # TIGHT ERROR

    # Create a function to get the tight error for a node
    def get_dist_tight_error(node):
        return errors.tight_error(
            node,
            eps,
            omega,
            num_branching_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U1, E1, Vh1 = svd.hasvd(tree, get_dist_hasvd_block, local_eps=get_dist_tight_error)
    duration = time.time() - start
    print("Tight Error:", np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1))
    print("Time of SVD:", duration)

    # INCREMENTAL HASVD
    print("\u0332".join("Incremental HASVD"))
    # Create a tree for the incremental HASVD
    tree = trees.two_level_bidir_inc_hasvd_tree(M, N, 1, 0, (m, n))
    # trees.draw_nxgraph(tree)
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
        return errors.naive_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U2, E2, Vh2 = svd.hasvd(tree, get_inc_hasvd_block, local_eps=get_inc_naive_error)
    duration = time.time() - start
    print("Naive Error:", np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2))
    print("Time of SVD:", duration)

    # TIGHT ERROR
    # Create a function to get the tight error for a node
    def get_inc_tight_error(node):
        return errors.tight_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U2, E2, Vh2 = svd.hasvd(tree, get_inc_hasvd_block, local_eps=get_inc_tight_error)
    duration = time.time() - start
    print("Tight Error:", np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2))
    print("Time of SVD:", duration)
# %%
