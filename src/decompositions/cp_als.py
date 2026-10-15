import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np

def frobenius_norm(tensor):
    """
    Compute the Frobenius norm of a tensor.
    
    The Frobenius norm is the square root of the sum of squares of all elements.
    ||X||_F = sqrt(sum of all X[i,j,k]^2)
    
    Parameters:
    -----------
    tensor : array-like
        Input tensor of any dimension
        
    Returns:
    --------
    float
        Frobenius norm of the tensor
    """
    return np.sqrt(np.sum(tensor ** 2))

def frobenius_error(original, reconstructed):
    """
    Compute the Frobenius norm of the difference between two tensors.
    This measures the reconstruction error.
    
    ||X - X_reconstructed||_F
    
    Parameters:
    -----------
    original : array-like
        Original tensor
    reconstructed : array-like
        Reconstructed tensor (must have same shape as original)
        
    Returns:
    --------
    float
        Frobenius norm of the error
    """
    error = original - reconstructed
    return frobenius_norm(error)

def relative_frobenius_error(original, reconstructed):
    """
    Compute the relative Frobenius error between two tensors.
    
    ||X - X_reconstructed||_F / ||X||_F
    
    Parameters:
    -----------
    original : array-like
        Original tensor
    reconstructed : array-like
        Reconstructed tensor (must have same shape as original)
        
    Returns:
    --------
    float
        Relative Frobenius error (0 = perfect reconstruction, 1 = 100% error)
    """
    error_norm = frobenius_error(original, reconstructed)
    original_norm = frobenius_norm(original)
    if original_norm == 0:
        return 0.0
    return error_norm / original_norm

# Example tensor (3x4x2)
X = tl.tensor(np.random.rand(2, 2, 2))

# Perform CP decomposition with rank = 1
print("Performing CP decomposition with rank = 1")
factors = parafac(X, rank=1)
print("Original tensor:\n", X)
# The result is a Kruskal tensor object, access factor matrices via .factors
A, B, C = factors.factors

print("Factor A (mode 1):\n", A)
print("Factor B (mode 2):\n", B)
print("Factor C (mode 3):\n", C)

# Reconstruct tensor from the decomposition
X_reconstructed = tl.cp_to_tensor(factors)

print("\nReconstructed tensor shape:", X_reconstructed.shape)

Y = tl.cp_to_tensor(factors)
print("Reconstructed tensor:\n", Y)

# Compute Frobenius norm metrics
print("\n" + "="*50)
print("Frobenius Norm Analysis:")
print("="*50)
print(f"Frobenius norm of original tensor: {frobenius_norm(X):.6f}")
print(f"Frobenius norm of reconstructed tensor: {frobenius_norm(X_reconstructed):.6f}")
print(f"Frobenius error (||X - X_reconstructed||_F): {frobenius_error(X, X_reconstructed):.6f}")
print(f"Relative Frobenius error: {relative_frobenius_error(X, X_reconstructed):.6f}")
print(f"Reconstruction accuracy: {(1 - relative_frobenius_error(X, X_reconstructed)) * 100:.2f}%")