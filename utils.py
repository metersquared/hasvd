import numpy as np
import scipy.linalg as scla

# Matrix generators


def random_matrix(
    n: int, m: int, r: int, rng: np.random.Generator, condition_number: float = None
):
    """
    Create a random n by m matrix of rank r with optional control over the condition number.

    Parameters
    ----------
    n : int
        Row size
    m : int
        Column size
    r : int
        Matrix rank
    rng : np.random.Generator
        Random number generator
    condition_number : float, optional
        Desired condition number (ratio of largest to smallest singular value).
        If None, singular values will be uniform.

    Returns
    -------
    np.ndarray
        Random n by m matrix of rank r
    """
    # Generate random orthonormal matrices U and V using SVD
    U, _, _ = np.linalg.svd(rng.random(size=(n, n)))
    V, _, _ = np.linalg.svd(rng.random(size=(m, m)))

    # Create singular values
    if condition_number:
        singular_values = np.geomspace(1, 1 / condition_number, num=r)
    else:
        singular_values = np.ones(r)

    # Construct the diagonal matrix
    S = np.zeros((n, m))
    S[:r, :r] = np.diag(singular_values)

    # Generate the rank-r matrix with specified conditioning
    A = U @ S @ V[:m, :].T

    return A


def array_to_hankel(
    array: np.ndarray,
    shape: tuple[int, int],
    block_coord: tuple[int, int] = None,
    block_shape: tuple[int, int] = None,
):
    n, m = shape

    if block_coord is None and block_shape is None:
        # Full Hankel matrix
        assert len(array) == n + m - 1
        first_column = array[:n]
        last_row = array[n - 1 :]
    else:
        i, j = block_coord
        p, q = block_shape
        assert i + p <= n and j + q <= m, "Block goes out of matrix bounds"
        offset = i + j
        block_len = p + q - 1
        block_array = array[offset : offset + block_len]
        first_column = block_array[:p]
        last_row = block_array[p - 1 :]

    return scla.hankel(first_column, last_row)


def hankel_to_array(a: np.ndarray):
    n = a.shape[0]
    m = a.shape[1]

    array = np.zeros(n + m - 1)

    for i in range(n):
        array[i] = a[i, 0]

    for i in range(1, m):
        array[i + n - 1] = a[n - 1, i]

    return array


def random_hankel(n: int, m: int, rng: np.random.Generator, low_rank=True):
    array = rng.random(size=n + m - 1)
    if low_rank:
        array = array / np.exp(np.arange(n + m - 1))
    return array_to_hankel(array, (n, m))


def blocks_to_hankel(
    M: int, N: int, blocks: list[np.ndarray], block_shape: tuple[int, int]
):
    """
    Create a block Hankel matrix from a list of blocks.

    Parameters
    ----------
    M : int
        Number of rows in the block Hankel matrix.
    N : int
        Number of columns in the block Hankel matrix.
    blocks : list[np.ndarray]
        List of blocks to be arranged in the Hankel structure.
    block_shape : tuple[int, int]
        Shape of each block.

    Returns
    -------
    np.ndarray
        Block Hankel matrix.
    """
    assert len(blocks) == M + N - 1

    # Create an empty array for the block Hankel matrix
    hankel_matrix = np.zeros((M * block_shape[0], N * block_shape[1]))

    for i in range(M):
        for j in range(N):
            hankel_matrix[
                i * block_shape[0] : (i + 1) * block_shape[0],
                j * block_shape[1] : (j + 1) * block_shape[1],
            ] = blocks[i + j]

    return hankel_matrix


def blocks_to_toeplitz(
    M: int, N: int, blocks: list[np.ndarray], block_shape: tuple[int, int]
):
    """
    Create a block Toeplitz matrix from a list of blocks.

    Parameters
    ----------
    M : int
        Number of rows in the block Toeplitz matrix.
    N : int
        Number of columns in the block Toeplitz matrix.
    blocks : list[np.ndarray]
        List of blocks to be arranged in the Toeplitz structure.
    block_shape : tuple[int, int]
        Shape of each block.

    Returns
    -------
    np.ndarray
        Block Toeplitz matrix.
    """
    assert len(blocks) == M + N - 1

    # Create an empty array for the block Toeplitz matrix
    toeplitz_matrix = np.zeros((M * block_shape[0], N * block_shape[1]))

    for i in range(M):
        for j in range(N):
            toeplitz_matrix[
                i * block_shape[0] : (i + 1) * block_shape[0],
                j * block_shape[1] : (j + 1) * block_shape[1],
            ] = blocks[j - i]

    return toeplitz_matrix


def array_to_toeplitz(
    array: np.ndarray,
    shape: tuple[int, int],
    block_coord: tuple[int, int] = None,
    block_shape: tuple[int, int] = None,
):
    n = shape[0]
    m = shape[1]
    assert n + m - 1 == len(array)

    if block_coord is None and block_shape is None:
        first_column = np.zeros(n)
        first_row = np.zeros(m)
    else:
        assert block_shape[0] - 1 + block_coord[0] < n
        assert block_shape[1] - 1 + block_coord[1] < m
        array = array[n - block_shape[0] - block_coord[0] + block_coord[1] :]
        n = block_shape[0]
        m = block_shape[1]
        first_column = np.zeros(n)
        first_row = np.zeros(m)

    for i in range(n):
        first_column[n - 1 - i] = array[i]

    for i in range(m):
        first_row[i] = array[i + n - 1]

    return scla.toeplitz(first_column, first_row)


def toeplitz_to_array(a: np.ndarray):
    n = a.shape[0]
    m = a.shape[1]

    array = np.zeros(n + m - 1)

    for i in range(n):
        array[n - 1 - i] = a[i, 0]

    for i in range(1, m):
        array[i + n - 1] = a[0, i]

    return array


def random_toeplitz(n: int, m: int, rng: np.random.Generator, low_rank=True):
    array = rng.random(size=n + m - 1)
    if low_rank:
        array = array / np.exp(np.arange(n + m - 1))
    return array_to_toeplitz(array, (n, m))


block_diag = scla.block_diag

# SVD


def method_of_snapshots(
    A: np.ndarray, full_matrices=False, truncate_tol=np.finfo(float).eps
):
    """Method of snapshots method to compute SVD

    Parameters
    ----------
    A : np.ndarray
        Matrix to compute
    truncate : bool, optional
        Economic method, by default False
    truncate_tol : _type_, optional
        Economic method tolerance, by default 1e-16

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        Right singular vectors U, Sequence of singular values E, Left singular vectors Vh
    """
    B = A.T @ A
    E, V = np.linalg.eigh(B)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(E)[::-1]
    E = E[sorted_indices]
    V = V[:, sorted_indices]

    # Determine truncation index based on the sum of smallest eigenvalues
    cumulative_sum = np.sqrt(np.cumsum(E[::-1])[::-1])  # Reverse cumulative sum
    truncation_index = np.searchsorted(
        cumulative_sum <= truncate_tol, True, side="left"
    )

    # Truncate small eigenvalues
    if not full_matrices:
        valid_indices = np.arange(len(E)) < truncation_index
        V = V[:, valid_indices]
        E = E[valid_indices]

    # Compute left singular vectors
    safe_eigenvalues = np.maximum(E, np.finfo(float).eps)
    scaling_factors = 1.0 / np.sqrt(safe_eigenvalues)
    U = A @ V @ np.diag(scaling_factors)

    return U, np.sqrt(safe_eigenvalues), V.T


def svd_with_tol(A: np.ndarray, full_matrices=False, truncate_tol=np.finfo(float).eps):
    """
    Wrapper for np.linalg.svd with truncation based on a tolerance.

    Parameters
    ----------
    A : np.ndarray
        Matrix to compute SVD of.
    full_matrices : bool, optional
        Whether to compute full or reduced SVD, by default False.
    truncate_tol : float, optional
        Tolerance for truncating small singular values, by default machine epsilon.

    Returns
    -------
    U : np.ndarray
        Left singular vectors.
    s : np.ndarray
        Singular values.
    Vh : np.ndarray
        Right singular vectors (transposed).
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=full_matrices)
    cumulative_sum = np.sqrt(np.cumsum(np.square(s[::-1]))[::-1])
    truncation_index = np.searchsorted(
        cumulative_sum <= truncate_tol, True, side="left"
    )

    if not full_matrices:
        valid = np.arange(len(s)) < truncation_index
        U = U[:, valid]
        s = s[valid]
        Vh = Vh[valid, :]
    return U, s, Vh


def dist_hasvd(A: np.ndarray, partitions: int, truncate=False, truncate_tol=1e-16):
    width = int(A.shape[1] / partitions)
    children = [A[:, i * width : (i + 1) * width] for i in range(partitions)]
    children_svd = []

    for child in children:
        children_svd.append(
            method_of_snapshots(
                child, full_matrices=truncate, truncate_tol=truncate_tol
            )
        )

    weighted_aggregated_U = np.concatenate(
        ([child[0] @ np.diag(child[1]) for child in children_svd]), axis=1
    )

    U, E, Vh = method_of_snapshots(
        weighted_aggregated_U, full_matrices=truncate, truncate_tol=truncate_tol
    )

    Vh = Vh @ block_diag(*[child[2] for child in children_svd])

    return U, E, Vh


import asyncio
from collections import defaultdict
import logging
from pymor.tools.random import spawn_rng
from threading import Thread


from pymor.algorithms.hapod import LifoExecutor, FakeExecutor, std_local_eps


def hasvd(
    tree,
    snapshots,
    local_eps,
    svd_method=svd_with_tol,
    executor=None,
    eval_snapshots_in_executor=False,
):
    """HASVD

    Parameters
    ----------
    tree : Tree <hasvd_Node>
        HASVD tree.
    snapshots : Map
        Snapshot matrix map
    local_eps : _type_
        _description_
    svd_method : Function, optional
        The method of SVD to compute matrix, by default np.linalg.svd
    executor : _type_, optional
        _description_, by default None
    eval_snapshots_in_executor : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    logger = logging.getLogger("hierarchical hasvd")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    node_finished_events = defaultdict(asyncio.Event)

    async def hasvd_step(node):

        if node.after:
            await asyncio.wait(
                [
                    asyncio.create_task(node_finished_events[a].wait())
                    for a in node.after
                ]
            )

        if node.children:
            results = await asyncio.gather(
                *(asyncio.create_task(hasvd_step(c)) for c in node.children)
            )

            U_parts, svals_parts, Vh_parts = zip(*results)

            if node.direction == 0:
                A = np.hstack([u * s for u, s in zip(U_parts, svals_parts)])
            elif node.direction == 1:
                A = np.hstack([v.T * s for v, s in zip(Vh_parts, svals_parts)])

        else:
            eps = local_eps(node)
            # logger.info(f'Obtaining snapshots for node {node.tag or ""} ...')

            if eval_snapshots_in_executor:
                A = await executor.submit(snapshots, node)
            else:
                A = snapshots(node)
            U, svals, Vh = svd_method(A, full_matrices=False, truncate_tol=eps)
            return U, svals, Vh

        # snap_count = sum(len(s) for s in svals_parts)
        eps = local_eps(node)
        if eps:
            # logger.info(f'Computing SVD at node {node.tag or ""} ...')
            if node.direction == 0:
                U, svals, Vh = svd_method(A, full_matrices=False, truncate_tol=eps)
                Vh = Vh @ scla.block_diag(*Vh_parts)
            elif node.direction == 1:
                V, svals, Uh = svd_method(A, full_matrices=False, truncate_tol=eps)
                U = scla.block_diag(*U_parts) @ Uh.T
                Vh = V.T
        else:
            svals = np.ones(A.shape[1])

        if node.tag is not None:
            node_finished_events[node.tag].set()
        return U, svals, Vh

    if executor is not None:
        executor = LifoExecutor(executor)
    else:
        executor = FakeExecutor

    def main():
        nonlocal result
        result = asyncio.run(hasvd_step(tree))

    result = None
    hasvd_thread = Thread(target=spawn_rng(main))
    hasvd_thread.start()
    hasvd_thread.join()

    return result


# Trees

from pymor.algorithms.hapod import Node


class hasvd_Node(Node):
    """docstring for hasvd_Node."""

    def __init__(
        self,
        shape: tuple[int, int] = None,
        direction=0,
        tag=None,
        parent=None,
        after=None,
    ):
        super().__init__(tag=tag, parent=parent, after=after)
        self.direction = direction
        self.m = shape[0]
        self.n = shape[1]
        self.root = parent.root if parent else self

    def add_child(self, direction=0, tag=None, after=None, **kwargs):
        return hasvd_Node(
            direction=direction, tag=tag, parent=self, after=after, **kwargs
        )

    @property
    def depth(self):
        return super().depth

    @property
    def is_leaf(self):
        return super().is_leaf

    @property
    def is_root(self):
        return super().is_root

    def traverse(self, return_level=False):
        return super().traverse(return_level)

    def __str__(self):
        lines = []
        for node, level in self.traverse(True):
            line = ""
            if node.parent:
                p = node.parent
                while p.parent:
                    if p.parent.children.index(p) + 1 == len(p.parent.children):
                        line = "  " + line
                    else:
                        line = "| " + line
                    p = p.parent
                match node.direction:
                    case 0:
                        line += "+-o"
                    case 1:
                        line += "+-r"
                    case 2:
                        line += "+-c"
            else:
                match node.direction:
                    case 0:
                        line += "o"
                    case 1:
                        line += "r"
                    case 2:
                        line += "c"
            if node.tag is not None:
                line += f" {node.tag}"
            if node.after:
                line += f' (after {",".join(str(a) for a in node.after)})'
            lines.append(line)
        return "\n".join(lines)


def inc_hasvd_tree(steps):
    tree = node = hasvd_Node()
    for step in range(steps)[::-1]:
        # add leaf node for a new snapshot
        node.add_child(tag=step, after=(step - 1,) if step > 0 else None)
        if step > 0:
            # add node for the previous POD step
            node = node.add_child()
    return tree


def dist_hasvd_tree(num_slices, arity=None, direction=0):
    tree = hasvd_Node(tag="root", direction=direction)
    if arity is None:
        arity = num_slices

    def add_children(node, slices):
        if len(slices) > arity:
            sub_slices = np.array_split(slices, arity)
            for s in sub_slices:
                if len(s) > 1:
                    child = node.add_child()
                    add_children(child, s)
                else:
                    child = node.add_child(tag=s.item())
        else:
            for s in slices:
                node.add_child(tag=s)

    add_children(tree, np.arange(num_slices))

    return tree


def two_level_bidir_dist_hasvd_tree(
    num_outer_slices: int,
    num_inner_slices: int,
    outer_direction: int = 0,
    inner_direction: int = 1,
    block_shape: tuple[int, int] = None,
):
    """
    Build a two-level hierarchical HASVD tree with full control over slice partitioning and directions.

    Parameters
    ----------
    num_outer_slices : int
        Number of top-level groups (outer partitions).
    num_inner_slices : int
        Number of inner slices per outer group (inner partitions).
    outer_direction : int, optional
        Aggregation direction at the outer level (default: 0).
    inner_direction : int, optional
        Aggregation direction at the inner level (default: 1).

    Returns
    -------
    hasvd_Node
        Root of the constructed HASVD tree.
    """
    total_slices = num_outer_slices * num_inner_slices
    root = hasvd_Node(
        tag="root",
        direction=outer_direction,
        shape=(num_outer_slices * block_shape[0], num_inner_slices * block_shape[1]),
    )

    # Inner nodes get tags 0, 1, ..., total_slices-1
    # Outer nodes get tags total_slices, total_slices+1, ...
    outer_tag = total_slices + 1
    for outer_idx in range(num_outer_slices):
        outer_node = root.add_child(
            tag=outer_tag,
            direction=inner_direction,
            shape=(block_shape[0], num_inner_slices * block_shape[1]),
        )
        outer_tag += 1
        for inner_idx in range(num_inner_slices):
            inner_tag = outer_idx * num_inner_slices + inner_idx + 1
            outer_node.add_child(tag=inner_tag, shape=(block_shape[0], block_shape[1]))

    return root


def two_level_bidir_inc_hasvd_tree(
    num_outer_slices: int,
    num_inner_slices: int,
    outer_direction: int = 0,
    inner_direction: int = 1,
    block_shape: tuple[int, int] = None,
):
    """
    Build a two-level hierarchical HASVD tree with full control over slice partitioning and directions.

    Parameters
    ----------
    num_outer_slices : int
        Number of top-level groups (outer partitions).
    num_inner_slices : int
        Number of inner slices per outer group (inner partitions).
    outer_direction : int, optional
        Aggregation direction at the outer level (default: 0).
    inner_direction : int, optional
        Aggregation direction at the inner level (default: 1).

    Returns
    -------
    hasvd_Node
        Root of the constructed HASVD tree.
    """
    total_leaves = num_outer_slices * num_inner_slices
    root = hasvd_Node(
        tag="root",
        direction=outer_direction,
        shape=(num_outer_slices * block_shape[0], num_inner_slices * block_shape[1]),
    )

    leaf_tag = 1
    outer_tag = total_leaves + 1
    parent_node = root

    for outer_idx in range(num_outer_slices):
        outer_node = parent_node.add_child(
            tag=outer_tag,
            direction=inner_direction,
            shape=(block_shape[0], num_inner_slices * block_shape[1]),
        )
        outer_tag += 1
        for inner_idx in range(num_inner_slices):
            outer_node.add_child(tag=leaf_tag, shape=(block_shape[0], block_shape[1]))
            leaf_tag += 1
        if outer_idx < num_outer_slices - 2:
            merge_node = parent_node.add_child(
                tag=outer_tag,
                direction=outer_direction,
                shape=(
                    (num_outer_slices - 1 - outer_idx) * block_shape[0],
                    num_inner_slices * block_shape[1],
                ),
            )
            outer_tag += 1
            parent_node = merge_node
    return root


# Graphs and trees

import networkx as nx
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


def random_kary_tree(n, k):
    nxG = nx.Graph()
    node_count = 1
    nxG.add_node(r"$\rho$")
    parent_count = 0
    leaf_nodes = [r"$\rho$"]
    while parent_count < n:
        for leaf_node in leaf_nodes:
            child_count = rng.choice(np.append(0, np.arange(2, k + 1)))
            for i in range(child_count):
                node_count = node_count + 1
                nxG.add_node(node_count)
                nxG.add_edge(node_count, leaf_node)
                leaf_nodes.append(node_count)
            if child_count > 0:
                parent_count = parent_count + 1
                leaf_nodes.remove(leaf_node)
            if parent_count == n:
                break
    pos = nx.nx_agraph.graphviz_layout(
        nxG,
        prog="dot",
        args=f"-Groot={1}",
    )
    return nxG, pos, leaf_nodes


def graphviz_for_tree(nxG: nx.Graph):
    return nx.nx_agraph.graphviz_layout(
        nxG,
        prog="dot",
        args=f"-Groot={1}",
    )


def draw_nxgraph(root: hasvd_Node, node_size=1000):
    nxG = nx.Graph()

    options = {"node_color": "white", "edgecolors": "black", "node_size": node_size}

    optionsRow = {
        "node_color": "tab:red",
        "edgecolors": "black",
        "node_shape": "s",
        "node_size": node_size,
    }

    optionsCol = {
        "node_color": "tab:blue",
        "edgecolors": "black",
        "node_shape": "D",
        "node_size": node_size,
    }

    leafNodes = []
    columnNodes = []
    rowNodes = []

    # Specify the root node
    root_node = root.tag

    # Add edges (example)
    for node in root.traverse():
        nxG.add_node(node.tag)
        if node.parent and node.tag != root.tag:
            nxG.add_edge(node.parent.tag, node.tag)
        match node.direction:
            case 0:
                leafNodes.append(node.tag)
            case 1:
                rowNodes.append(node.tag)
            case 2:
                columnNodes.append(node.tag)

    # Generate the layout using graphviz_layout
    pos = nx.nx_agraph.graphviz_layout(
        nxG,
        prog="dot",
        args=f"-Groot={root_node}",
    )

    # Draw the graph
    plt.figure(figsize=(20, 20))
    plt.axis("off")
    plt.rcParams["text.usetex"] = True
    nx.draw(nxG, pos)
    nx.draw_networkx_nodes(nxG, pos, nodelist=leafNodes, **options)
    nx.draw_networkx_nodes(nxG, pos, nodelist=rowNodes, **optionsRow)
    nx.draw_networkx_nodes(nxG, pos, nodelist=columnNodes, **optionsCol)
    nx.draw_networkx_edges(nxG, pos, edgelist=nxG.edges())
    nx.draw_networkx_labels(
        nxG,
        pos,
        {n: n for n in columnNodes + rowNodes if n in pos},
        font_size=24,
        font_color="white",
    )
    nx.draw_networkx_labels(
        nxG,
        pos,
        {n: n for n in leafNodes if n in pos},
        font_size=24,
        font_color="black",
    )
