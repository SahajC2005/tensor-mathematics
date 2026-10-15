from cp import video_tensor
import tensorly as tl
from tensorly.decomposition import parafac

tl.set_backend('numpy')

rank = 20    # choose rank suitable for your video
factors = parafac(
    video_tensor,
    rank=rank,
    n_iter_max=200,     
    init='svd',         
    tol=1e-6,           
    normalize_factors=True
)

A, B, C = factors.factors