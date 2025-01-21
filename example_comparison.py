import numpy as np
import numpy.linalg as la
import utils
import time

n = 1000
truncate = True
tol = 1e-16
print("Truncation tolerance:", tol)

rng = np.random.default_rng(12)
# A = utils.random_matrix(4 * n, 4 * n, 2, rng, condition_number=1.0)
A = utils.random_hankel(4 * n, 4 * n, rng)

print("Matrix rank:", la.matrix_rank(A))
print("Matrix condition:", la.cond(A))


# "Normal" SVD
print("\u0332".join("Normal SVD"))

start_time = time.time()
U, E, Vh = la.svd(A, full_matrices=False)
print("--- %s seconds ---" % (time.time() - start_time))

print("Norm:", la.norm(A - U @ np.diag(E) @ Vh))

# Method of snapshots
print("\u0332".join("Method of snapshots"))


def method_of_snapshots(A: np.ndarray, truncate=False, truncate_tol=1e-16):
    B = A.T @ A
    E, V = np.linalg.eigh(B)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(E)[::-1]
    E = E[sorted_indices]
    V = V[:, sorted_indices]

    # Truncate small eigenvalues
    if truncate:
        valid_indices = E > max(truncate_tol, 0)
        V = V[:, valid_indices]
        E = E[valid_indices]

    # Compute left singular vectors
    safe_eigenvalues = np.maximum(E, np.finfo(float).eps)
    scaling_factors = 1.0 / np.sqrt(safe_eigenvalues)
    U = A @ V @ np.diag(scaling_factors)

    return U, np.sqrt(safe_eigenvalues), V.T


start_time = time.time()
U, E, Vh = method_of_snapshots(A, truncate=truncate, truncate_tol=tol)
print("--- %s seconds ---" % (time.time() - start_time))

print("Norm I-Vh@V:", la.norm(np.identity(len(E)) - Vh @ Vh.T))
print("Norm I-Uh@U:", la.norm(np.identity(len(E)) - U.T @ U))

print("Norm after truncation:", la.norm(A - U @ np.diag(E) @ Vh))

# Distributed HASVD
print("\u0332".join("Distributed even HASVD"))


def dist_hasvd(A: np.ndarray, partitions: int, truncate=False, truncate_tol=1e-16):
    width = int(A.shape[1] / partitions)
    children = [A[:, i * width : (i + 1) * width] for i in range(partitions)]
    children_svd = []

    for child in children:
        children_svd.append(
            method_of_snapshots(child, truncate=truncate, truncate_tol=truncate_tol)
        )

    weighted_aggregated_U = np.concatenate(
        ([child[0] @ np.diag(child[1]) for child in children_svd]), axis=1
    )

    U, E, Vh = method_of_snapshots(
        weighted_aggregated_U, truncate=truncate, truncate_tol=truncate_tol
    )

    Vh = Vh @ utils.block_diag(*[child[2] for child in children_svd])

    return U, E, Vh


partitions = 4
print("Partitions:", partitions)
start_time = time.time()
Udist, Edist, Vhdist = dist_hasvd(A, partitions, truncate=truncate, truncate_tol=tol)
print("--- %s seconds ---" % (time.time() - start_time))

print("Norm I-Vh@V:", la.norm(np.identity(len(Edist)) - Vhdist @ Vhdist.T))
print("Norm I-Uh@U:", la.norm(np.identity(len(Edist)) - Udist.T @ Udist))

print("Norm after truncation:", la.norm(A - Udist @ np.diag(Edist) @ Vhdist))


# Diagonal solving
print("\u0332".join("Diagonal HASVD"))

start_time = time.time()
width = int(A.shape[1] / 2)

h = [
    A[:width, :width],
    A[:width, width : 2 * width],
    A[width : 2 * width, width : 2 * width],
]

h_svd = [method_of_snapshots(hi) for hi in h]

H_SVD = []

H_2 = method_of_snapshots(
    np.block(
        [
            [
                np.zeros((width, width)),
                h_svd[1][0] @ np.diag(h_svd[1][1]),
            ],
            [h_svd[1][0] @ np.diag(h_svd[1][1]), np.zeros((width, width))],
        ]
    )
)

v_2h = H_2[2] @ np.block(
    [
        [h_svd[1][2], np.zeros((width, width))],
        [
            np.zeros((width, width)),
            h_svd[1][2],
        ],
    ]
)

H_1 = method_of_snapshots(
    np.block(
        [
            [h_svd[0][0] @ np.diag(h_svd[0][1]), np.zeros((width, width))],
            [
                np.zeros((width, width)),
                h_svd[2][0] @ np.diag(h_svd[2][1]),
            ],
        ]
    )
)

v_1h = H_1[2] @ np.block(
    [
        [h_svd[0][2], np.zeros((width, width))],
        [
            np.zeros((width, width)),
            h_svd[2][2],
        ],
    ]
)

BIG_SVD = method_of_snapshots(
    H_1[0] @ np.diag(H_1[1]) + H_2[0] @ np.diag(H_2[1]) @ v_2h @ v_1h.T
)

VH = BIG_SVD[2] @ v_1h
print("--- %s seconds ---" % (time.time() - start_time))

print("Norm Diagonal SVD:", la.norm(A - BIG_SVD[0] @ np.diag(BIG_SVD[1]) @ VH))

# Trees

# print(utils.inc_tree(5))
