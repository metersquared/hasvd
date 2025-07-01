import numpy as np
import scipy.linalg as scla
from hasvd.utils.trees import hasvd_Node

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
    cumulative_sum = np.sqrt(
        np.cumsum(np.clip(E, a_min=0, a_max=None)[::-1])[::-1]
    )  # Reverse cumulative sum
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
    track_ranks=False,
):
    """Hierarchical Approximate SVD with optional rank tracking

    If track_ranks is True, returns a tuple (U, svals, Vh, node_rank_map),
    where node_rank_map maps each node tag to a dict of its children's ranks.
    """
    logger = logging.getLogger("hierarchical_hasvd")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(ch)

    # Map to track ranks per node
    node_rank_map = {} if track_ranks else None

    node_finished_events = defaultdict(asyncio.Event)

    async def hasvd_step(node):
        # wait for dependencies
        if node.after:
            await asyncio.wait(
                [
                    asyncio.create_task(node_finished_events[a].wait())
                    for a in node.after
                ]
            )

        if node.children:
            # recurse into children
            results = await asyncio.gather(
                *(asyncio.create_task(hasvd_step(c)) for c in node.children)
            )
            # results are tuples (U, svals, Vh[, node_rank_map])
            # unpack based on track_ranks
            if track_ranks:
                # child_results include their maps; ignore their maps here
                U_parts, svals_parts, Vh_parts, _ = zip(*results)
            else:
                U_parts, svals_parts, Vh_parts = zip(*results)

            # assemble matrix A
            if node.direction == 0:
                A = np.hstack([u * s for u, s in zip(U_parts, svals_parts)])
            else:
                A = np.hstack([v.T * s for v, s in zip(Vh_parts, svals_parts)])

        else:
            # leaf: compute SVD directly
            eps = local_eps(node)
            if eval_snapshots_in_executor:
                A = await executor.submit(snapshots, node)
            else:
                A = snapshots(node)
            U_parts, svals_parts, Vh_parts = svd_method(
                A, full_matrices=False, truncate_tol=eps
            )

        # for non-root nodes or after assembling A
        eps = local_eps(node)
        if eps:
            if node.children:
                if node.direction == 0:
                    U, svals, Vh = svd_method(A, full_matrices=False, truncate_tol=eps)
                    Vh = Vh @ scla.block_diag(*Vh_parts)
                else:
                    V, svals, Uh = svd_method(A, full_matrices=False, truncate_tol=eps)
                    U = scla.block_diag(*U_parts) @ Uh.T
                    Vh = V.T
            else:
                U, svals, Vh = U_parts, svals_parts, Vh_parts
        else:
            # no truncation requested
            U, svals, Vh = U_parts, svals_parts, Vh_parts

        # track this node's own rank
        if track_ranks and node is not None:
            node_rank_map[node] = len(svals)

        # signal completion
        if node.tag is not None:
            node_finished_events[node.tag].set()

        if track_ranks:
            return U, svals, Vh, node_rank_map
        else:
            return U, svals, Vh

    # executor setup
    if executor is not None:
        executor = LifoExecutor(executor)
    else:
        executor = FakeExecutor

    # run recursion in separate thread
    result = None

    def main():
        nonlocal result
        result = asyncio.run(hasvd_step(tree))

    hasvd_thread = Thread(target=spawn_rng(main))
    hasvd_thread.start()
    hasvd_thread.join()

    return result


def rank_analysis(tree: hasvd_Node, ranks):
    leaves_rank = [ranks[node] for node in tree.traverse() if node.is_leaf]
    non_leaves_rank = [
        ranks[node] for node in tree.traverse() if not node.is_leaf and not node.is_root
    ]
    if len(leaves_rank) > 0:
        print("\033[31mLeaf node rank analysis\033[0m")
        mean_rank = np.mean(leaves_rank)
        median_rank = np.median(leaves_rank)
        stdv_rank = np.std(leaves_rank)
        mode_rank = max(set(leaves_rank), key=leaves_rank.count)
        max_rank = max(leaves_rank)
        min_rank = min(leaves_rank)
        print("Interval:", min_rank, "-", max_rank)
        print("Mean:", mean_rank)
        print("Median:", median_rank)
        print("Std var:", stdv_rank)
        print("Mode:", mode_rank)

    if len(non_leaves_rank) > 0:
        print("\033[31mNon-leaf node rank analysis\033[0m")
        mean_rank = np.mean(non_leaves_rank)
        median_rank = np.median(non_leaves_rank)
        stdv_rank = np.std(non_leaves_rank)
        mode_rank = max(set(non_leaves_rank), key=non_leaves_rank.count)
        max_rank = max(non_leaves_rank)
        min_rank = min(non_leaves_rank)
        print("Interval:", min_rank, "-", max_rank)
        print("Mean:", mean_rank)
        print("Median:", median_rank)
        print("Std var:", stdv_rank)
        print("Mode:", mode_rank)
