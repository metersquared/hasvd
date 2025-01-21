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
    n = shape[0]
    m = shape[1]
    assert n + m - 1 == len(array)

    if block_coord is None and block_shape is None:
        first_column = np.zeros(n)
        last_row = np.zeros(m)
    else:
        assert block_shape[0] - 1 + block_coord[0] < n
        assert block_shape[1] - 1 + block_coord[1] < m
        n = block_shape[0]
        m = block_shape[1]
        first_column = np.zeros(n)
        last_row = np.zeros(m)
        array = array[block_coord[0] + block_coord[1] :]

    for i in range(n):
        first_column[i] = array[i]

    for i in range(m):
        last_row[i] = array[i + n - 1]

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

    # Truncate small eigenvalues
    if not full_matrices:
        valid_indices = E > max(truncate_tol, 0)
        V = V[:, valid_indices]
        E = E[valid_indices]

    # Compute left singular vectors
    safe_eigenvalues = np.maximum(E, np.finfo(float).eps)
    scaling_factors = 1.0 / np.sqrt(safe_eigenvalues)
    U = A @ V @ np.diag(scaling_factors)

    return U, np.sqrt(safe_eigenvalues), V.T


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

"""
def hasvd(
    tree,
    submatrix,
    local_eps,
    product=None,
    svd_method=np.linalg.svd,
    executor=None,
    eval_snapshots_in_executor=False,
):
    logger = getLogger("hasvd")

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
            modes, svals, snap_counts = zip(
                *await asyncio.gather(
                    *(spawn_rng(hasvd_step(c)) for c in node.children)
                )
            )
            for m, sv in zip(modes, svals):
                m.scal(sv)
            U = modes[0]
            for V in modes[1:]:
                U.append(V, remove_from_other=True)
            snap_count = sum(snap_counts)
        else:
            logger.info(f'Obtaining snapshots for node {node.tag or ""} ...')
            if eval_snapshots_in_executor:
                U = await executor.submit(submatrix, node)
            else:
                U = submatrix(node)
            snap_count = len(U)

        eps = local_eps(node, snap_count, len(U))
        if eps:
            logger.info("Computing intermediate POD ...")
            modes, svals = await executor.submit(
                svd_method, U, eps, not node.parent, product
            )
        else:
            modes, svals = U.copy(), np.ones(len(U))
        if node.tag is not None:
            node_finished_events[node.tag].set()
        return modes, svals, snap_count

    # wrap Executer to ensure LIFO ordering of tasks
    # this ensures that PODs of parent nodes are computed as soon as all input data
    # is available
    if executor is not None:
        executor = LifoExecutor(executor)
    else:
        executor = FakeExecutor

    # run new asyncio event loop in separate thread to not interfere with
    # already running event loops (e.g. jupyter)

    def main():
        nonlocal result
        result = asyncio.run(hasvd_step(tree))

    result = None
    hapod_thread = Thread(target=spawn_rng(main))
    hapod_thread.start()
    hapod_thread.join()
    return result

"""


def hasvd(
    tree,
    snapshots,
    local_eps,
    svd_method=np.linalg.svd,
    executor=None,
    eval_snapshots_in_executor=False,
):
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
            logger.info(f'Obtaining snapshots for node {node.tag or ""} ...')

            if eval_snapshots_in_executor:
                A = await executor.submit(snapshots, node)
            else:
                A = snapshots(node)

            U, svals, Vh = svd_method(A, full_matrices=False)
            return U, svals, Vh

        snap_count = sum(len(s) for s in svals_parts)
        eps = local_eps(node, snap_count, A.shape[1])
        if eps:
            logger.info(f'Computing SVD at node {node.tag or ""} ...')
            if node.direction == 0:
                U, svals, Vh = svd_method(A, full_matrices=False)
                Vh = Vh @ scla.block_diag(*Vh_parts)
            elif node.direction == 1:
                V, svals, Uh = svd_method(A, full_matrices=False)
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

    def __init__(self, direction=0, tag=None, parent=None, after=None):
        super().__init__(tag=tag, parent=parent, after=after)
        self.direction = direction

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
                if node.direction == 0:
                    line += "+-o"
                else:
                    line += "+-x"
            else:
                if node.direction == 0:
                    line += "o"
                else:
                    line += "x"
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
