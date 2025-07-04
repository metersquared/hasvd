import numpy as np
import scipy.linalg as scla

# Matrix generators


def random_matrix(m: int, n: int, r: int, rng: np.random.Generator):
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
    U, _, _ = np.linalg.svd(rng.random(size=(m, m)))
    V, _, _ = np.linalg.svd(rng.random(size=(n, n)))
    A = np.random.rand(n, n)

    max_dim = max(m, n)
    eps = np.finfo(np.float64).eps

    # Construct the diagonal matrix
    S = np.zeros((m, n))
    S[np.diag_indices(r)] = np.geomspace(1, max_dim * eps, r)

    # Generate the rank-r matrix with specified conditioning
    A = U @ S @ V.T

    return A


# Sequence generators


def lrf_sequence(rank, length, rng: np.random.Generator):

    # Convert roots to polynomial coefficients
    coeffs = rng.standard_normal(rank)
    coeffs /= np.linalg.norm(coeffs)

    # Random initial values (normalized)
    a = np.zeros(length)
    a[:rank] = rng.standard_normal(rank)
    # a[:rank] /= np.linalg.norm(a[:rank])

    for i in range(rank, length):
        a[i] = np.dot(coeffs, a[i - rank : i])
        # a[:i] /= np.linalg.norm(a[:i])

    a /= np.linalg.norm(a)

    return a


def fid_signal_sequence(rank, length, rng: np.random.Generator):

    # Signal coefficients
    coeffs = rng.standard_normal(rank)
    # coeffs /= np.linalg.norm(coeffs)

    # Damp factors
    tau = rng.uniform(0, 5.0, rank)

    # FID signals
    ts = np.arange(length)
    x = np.zeros(length)

    for i, t in enumerate(ts):
        x[i] = np.dot(coeffs, np.exp(-tau * t))

    return x


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


def random_hankel(
    n: int, m: int, rng: np.random.Generator, seq_generator="def", rank=20
):
    if seq_generator == "def":
        array = rng.random(size=n + m - 1)
        array = array / np.exp(np.arange(n + m - 1))
    elif seq_generator == "lrf":
        array = lrf_sequence(rank, n + m - 1, rng)
    elif seq_generator == "fid":
        array = fid_signal_sequence(rank, n + m - 1, rng)

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
