import numpy as np

def create_tensor():
    A = np.zeros((2,2,2), dtype=float)
    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                A[i1, i2, i3] = (i1+1) + (i2+1) + (i3+1)
    return A

tensor = create_tensor()
print("Original tensor:\n", tensor)

def savevalue(Value):
    Value1 = Value
    print("Saved value:", Value1)

def tt_svd(tensor, tol=1e-12):
    dims = tensor.shape
    d = len(dims)
    cores = []
    curr = tensor.copy().astype(float)
    r_prev = 1

    for k in range(d-1):
        n_k = dims[k]
        M = curr.reshape((r_prev * n_k, -1))
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
       #any smaller value than tol is ignored for the egien matrix which cotnains the important information
       
        r_k = np.sum(S > tol)
        if r_k == 0:
            r_k = 1
        
        U = U[:, :r_k]
        S = S[:r_k]
        Vt = Vt[:r_k, :]

        savevalue(U)
        savevalue(S)
        savevalue(Vt)

        core = U.reshape((r_prev, n_k, r_k))
        cores.append(core)
        curr = np.dot(np.diag(S), Vt)
        r_prev = r_k
        curr = curr.reshape((r_prev,) + dims[k+1:])

    last_core = curr.reshape((r_prev, dims[-1], 1))
    cores.append(last_core)
    return cores, d
    


# Run demo
A = create_tensor()
cores, d = tt_svd(A)  # Unpack the tuple return value

dims = A.shape
print(f"Number of dimensions: {d}")

print("Original tensor A:\n", A)
print("\nTT core shapes and values:")
for i, c in enumerate(cores):
    print(f"\nCore {i+1} shape: {c.shape}\n{c}")
