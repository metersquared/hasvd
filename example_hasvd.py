import utils
import numpy as np
import numpy.linalg as la
import time

rng = np.random.default_rng(12)
n = 100
i = 4
A = utils.random_hankel(i * n, i * n, rng)
print("Matrix rank:", la.matrix_rank(A))
print("Matrix condition:", la.cond(A))

# "Normal" SVD
print("\u0332".join("Normal SVD"))

start_time = time.time()
U, E, Vh = la.svd(A, full_matrices=False)
print("--- %s seconds ---" % (time.time() - start_time))

print("Norm:", la.norm(A - U @ np.diag(E) @ Vh))

# Method of snapshots
print("\u0332".join("Method of Snapshot"))

start_time = time.time()
U1, E1, Vh1 = utils.method_of_snapshots(A, full_matrices=False)
print("--- %s seconds ---" % (time.time() - start_time))

print("Norm:", la.norm(A - U1 @ np.diag(E1) @ Vh1))

# Distributed Horizontal HASVD
print("\u0332".join("Horizontal HASVD"))

array = utils.hankel_to_array(A)
partitions = 10


def snapshot1(node):
    return utils.array_to_hankel(
        array,
        A.shape,
        (0, node.tag * int(A.shape[1] / partitions)),
        (i * n, int(A.shape[1] / partitions)),
    )


tree1 = utils.dist_hasvd_tree(partitions)

start_time = time.time()
U2, E2, Vh2 = utils.hasvd(
    tree1, snapshot1, utils.std_local_eps, svd_method=utils.method_of_snapshots
)
print("--- %s seconds ---" % (time.time() - start_time))

print("Norm:", la.norm(A - U2 @ np.diag(E2) @ Vh2))

# Distributed Vertical HASVD
print("\u0332".join("Vertical HASVD"))

array = utils.hankel_to_array(A)
partitions = 10


def snapshot2(node):
    return utils.array_to_hankel(
        array,
        A.shape,
        (node.tag * int(A.shape[0] / partitions), 0),
        (int(A.shape[0] / partitions), i * n),
    )


tree2 = utils.dist_hasvd_tree(partitions, direction=1)

print(tree2)

start_time = time.time()
U3, E3, Vh3 = utils.hasvd(
    tree2, snapshot2, utils.std_local_eps, svd_method=utils.method_of_snapshots
)
print("--- %s seconds ---" % (time.time() - start_time))

print("Norm:", la.norm(A - U3 @ np.diag(E3) @ Vh3))
