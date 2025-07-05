import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
import numpy as np

M = 10
N = 10
m = 500
n = 500
rng = np.random.Generator(np.random.MT19937(42))
eps = 1.0e-3
omega = 0.1
rank =20
direction =0

A_array = matrix.fid_signal_sequence(20,M*m+N*n-1, rng)




def nodemap_hasvd(tree: trees.hasvd_Node):
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
    assert 
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
