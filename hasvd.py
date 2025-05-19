# %%
import asyncio, utils
import numpy as np
from collections import defaultdict
from pymor.core.logger import getLogger


def hasvd(
    tree: utils.hasvd_Node,
    matrix,
    local_eps,
    svd_method=utils.method_of_snapshots,
):
    """
    Perform the HASVD algorithm on a tree of nodes.

    Parameters
    ----------
    tree : utils.hasvd_Node
        The root node of the tree.
    snapshots : function
        A function that takes a node and returns a snapshot matrix.
    local_eps : float
        The local epsilon value for the SVD.
    svd_method : function, optional
        The SVD method to use. The default is utils.method_of_snapshots.

    Returns
    -------
    U : ndarray
        The left singular vectors.
    E : ndarray
        The singular values.
    Vh : ndarray
        The right singular vectors.
    """

    node_finished_events = defaultdict(
        asyncio.Event
    )  # Dictionary to store events for each node

    logger = getLogger("pymor.algorithms.hapod.hapod")


if __name__ == "__main__":

    M = 2
    N = 3
    m = 2
    n = 2
    rng = np.random.default_rng(12)

    # Create a random matrix with block size m x n with M xN blocks
    rng = np.random.default_rng(12)
    A = utils.random_hankel(m * M, n * N, rng)
    print("A:", A)
    print("Matrix rank:", np.linalg.matrix_rank(A))
    print("Matrix shape:", A.shape)

    A_array = utils.hankel_to_array(A)
    print("A_array:", A_array)

    tree = utils.two_level_bidir_dist_hasvd_tree(M, N, 0, 1)
    utils.draw_nxgraph(tree)

    # Create mapping of nodes to their corresponding matrix blocks
    node_to_block = {}
    for node in tree.traverse():

        # Check if the node is a leaf node
        if node.is_leaf:
            block_coord = (
                np.floor_divide(node.tag - 1, N),
                ((node.tag - 1) % N) * n,
            )
            block_shape = (m, n)
            print(
                f"Node {node.tag} block coordinates: {block_coord}, shape: {block_shape}"
            )
            # Get the block corresponding to the node
            block = utils.array_to_hankel(
                A_array,
                A.shape,
                block_coord,
                block_shape,
            )
            node_to_block[node] = block
    # Print the blocks for each node
    for node, block in node_to_block.items():
        print(f"Node {node.tag} block:\n{block}")


# %%
