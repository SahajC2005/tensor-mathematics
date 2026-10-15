import numpy as np
import matplotlib.pyplot as plt

def simple_tt_svd(tensor, tol=1e-12):
    """
    Simple Tensor-Train decomposition (TT-SVD) for beginners.
    Returns a list of TT cores.
    """
    dims = tensor.shape
    cores = []
    curr = tensor.copy().astype(float)  # current tensor to process
    r_prev = 1  # previous rank, start with 1
    value = 3
    for k in range(len(dims) - 1):
        n_k = dims[k]


        M = curr.reshape(r_prev * n_k, -1)

        # SVD decomposition
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Plot singular values with labeled axes
        plt.figure(figsize=(7, 4))
        plt.semilogy(S, marker='o')
        plt.xlabel('Singular value index')
        plt.ylabel('Singular value (log scale)')
        plt.title(f'TT-SVD step {k+1}: singular values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"Step {k+1} S shape: {S.shape}")
        # Decide how many singular values to keep
        r_k = np.sum(S > tol)
        r_k = max(r_k, 1)  # at least rank 1


        # Keep only top singular values
        U = U[:, :r_k]
        S = S[:r_k]
        Vt = Vt[:r_k, :]

        
        # Create TT core for this dimension
        core = U.reshape(r_prev, n_k, r_k)
        cores.append(core)

        # Prepare remaining tensor for next dimension
        curr = np.dot(np.diag(S), Vt)
        r_prev = r_k
        curr = curr.reshape((r_prev,) + dims[k+1:])

    # Last core
    last_core = curr.reshape(r_prev, dims[-1], 1)
    cores.append(last_core)

    return cores


A = np.array([[[1,2],[3,4]], [[5,6],[7,8]]], dtype=float)

cores = simple_tt_svd(A)

# Show TT core shapes
for i, c in enumerate(cores):
    print(f"Core {i+1} shape: {c.shape}")

