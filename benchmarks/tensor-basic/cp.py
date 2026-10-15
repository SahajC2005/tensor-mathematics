import cv2
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import sys
import matplotlib.pyplot as plt


video = cv2.VideoCapture("Vid.mov")

frame = []


while True:
    check, img = video.read()
    if not check:
        break

    img_resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    frame.append(gray)

video.release()
video_tensor = np.stack(frame, axis=2)


print(video_tensor.shape)
print(frame[0])

# Only run visualization and analysis when script is run directly, not when imported
if __name__ == "__main__":
    plt.imshow(frame[0], cmap='gray')
    plt.show()

    if 'debugpy' in sys.modules:
        print("Running in VS Code")

    factors = parafac(
        video_tensor,
        rank = 3,
        n_iter_max=100,     # ALS iterations
        init='svd',         
        tol=1e-6,           # stopping condition
        normalize_factors=True
    )


    A, B, C = factors.factors 

    print(A.shape, B.shape, C.shape)

    print("Vector A:", A)
    print("Vector B:",B)
    print("Vector C:",C)

    print("Number of C vectors in Cp decomposition:", len(C))
    print("Number of B vectors in Cp decomposition:", len(B))
    print("Number of A vectors in Cp decomposition:", len(A))



    Y = tl.cp_to_tensor(factors)
    print(Y.shape)
    print(Y[:, :, 0])

    plt.imshow(Y[:, :, 0], cmap='gray')
    plt.show()


    spatial_pattern_0 = np.outer(A[:,0], B[:,0])  # reconstruct 2D image

    plt.imshow(spatial_pattern_0, cmap='gray')
    plt.title("Spatial Pattern of Component 0")
    plt.show()


    spatial_pattern_1 = np.outer(A[:,1], B[:,1])  # reconstruct 2D image

    plt.imshow(spatial_pattern_1, cmap='gray')
    plt.title("Spatial Pattern of Component 1")
    plt.show()


    spatial_pattern_2 = np.outer(A[:,2], B[:,2])  # reconstruct 2D image

    plt.imshow(spatial_pattern_2, cmap='gray')
    plt.title("Spatial Pattern of Component 2")
    plt.show()


    error = np.linalg.norm((video_tensor - Y).ravel())
    relative_error = error / np.linalg.norm(video_tensor.ravel())
    print("Frobenius error:", error)
    print("Relative error:", relative_error)



