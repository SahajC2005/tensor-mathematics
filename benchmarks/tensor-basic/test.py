
import cv2
import cp
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd


frame = cp.frame
video = cp.video
video_tensor = cp.video_tensor

class decompose:
    def __init__(self):
        pass

    def process_video(self, video, frame):
        while True:
            check, img = video.read()
            if not check:
                break
            img_resized = cv2.resize(img, (64, 64))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            frame.append(gray)
        
        video.release()
        video_tensor = np.stack(frame, axis=2)
        return video_tensor

class rank:
    my_globalvarx = 10 
    def __init__(self, video_tensor):
        self.video_tensor = video_tensor
        self.rank_value = 3
        
    def parfac(self, my_globalvarx):
        factors = parafac(
            self.video_tensor,
            rank=my_globalvarx,
            n_iter_max=100,     # ALS iterations
            init='svd',         
            tol=1e-6,           # stopping condition
            normalize_factors=True
        )
        return factors

    def factors(self, my_globalvarx):
        factors = self.parfac(my_globalvarx)
        A, B, C = factors.factors
        return A, B, C 


class Error:
    Error_dict = {}  # Class variable shared across all instances
    
    def __init__(self, var, val, obj):
        self.video_tensor1 = var
        self.rank_value = val  # Store the rank value
        self.rank_obj = rank(self.video_tensor1)
        self.factors_obj = self.rank_obj.parfac(val)
        A, B, C = self.factors_obj.factors
        self.A = A
        self.B = B
        self.C = C
    
    def calculate(self):
        Y = tl.cp_to_tensor(self.factors_obj)
        error = np.linalg.norm((self.video_tensor1 - Y).ravel())
        relative_error = float(error / np.linalg.norm(self.video_tensor1.ravel()))  # Explicitly convert to float
        print("Frobenius error:", error)
        print("Relative error:", relative_error)
        Error.Error_dict[self.rank_value] = relative_error  # Use rank value as key
        return error, relative_error

for i,j in enumerate(range(1, 64)):
    error = Error(video_tensor, j, i)
    print(f"Rank: {j}, Error: {error.calculate()}")
    error.calculate()


plt.plot(Error.Error_dict.keys(), Error.Error_dict.values())
plt.xlabel("Rank")
plt.ylabel("Error")
plt.title("Error vs Rank")
plt.show()


# Convert dictionary values to float arrays and sort by error values
error_values = np.array(list(Error.Error_dict.values()), dtype=float)  # Error values (floats)
rank_values = np.array(list(Error.Error_dict.keys()), dtype=float)     # Rank values (as floats)

# Sort by error values for proper polynomial fitting
sort_idx = np.argsort(error_values)
x = error_values[sort_idx].reshape(-1, 1)  # Sorted error values
y = rank_values[sort_idx].reshape(-1, 1)   # Corresponding rank values

# Calculate gaps between consecutive data points
x_gaps = np.diff(x.flatten())  # Gap in error values between consecutive points

print("\n=== Data Point Gaps ===")
print(f"Number of gaps: {len(x_gaps)}")
print(f"\nError gaps (x-axis): min={x_gaps.min():.6f}, max={x_gaps.max():.6f}, mean={x_gaps.mean():.6f}")

# Plot
fig, axes = plt.subplots(1, 1, figsize=(12, 6))   # Wider + cleaner

padding = 0.02  # smaller padding for tighter zoom

axes.plot(
    x_gaps,
    marker='o',
    markersize=6,       # bigger points
    linewidth=2         # thicker line
)

axes.set_title('Gaps in Error Values (x-axis)', fontsize=16)
axes.set_xlabel('Data Point Index', fontsize=14)
axes.set_ylabel('Gap in Error', fontsize=14)
axes.grid(True, alpha=0.3)

# --- Y-axis zoom ---
axes.set_ylim(x_gaps.min() - padding, x_gaps.max() + padding)

# --- X-axis zoom (optional) ---
# axes.set_xlim(0, 20)  # Example: zoom into first 20 points

plt.tight_layout()
plt.show()

log_y = np.log(y)

# Fit linear regression: log(y) = slope*x + intercept
# This means: y = exp(slope*x + intercept) = exp(intercept) * exp(slope*x)
model = LinearRegression()
model.fit(x, log_y)

# Extract parameters
slope = model.coef_[0][0]      # coefficient from log(y) = slope*x + intercept
intercept = model.intercept_[0]  # intercept
a = np.exp(intercept)          # initial value (when x=0)
b = slope                      # rate parameter

print(f"a = {a:.3f}, b = {b:.3f}")
print(f"Model: y = {a:.3f} * exp({b:.3f} * x)")

# Predict values: y = exp(slope*x + intercept) = a * exp(b*x)
y_pred = np.exp(model.predict(x))

# Plot original data and fitted curve
plt.scatter(x, y, label='Actual Data')
plt.plot(x, y_pred, color='red', label='Exponential Fit')
plt.xlabel('Error')
plt.ylabel('Rank')
plt.title('Rank vs Error (Exponential Fit)')
plt.legend()
plt.show()

df = pd.DataFrame({
    "Rank": rank_values,
    "Error": error_values
})

min_error = df["Error"].min()
print("Minimum Error:", min_error)
min_rank = df[df["Error"] == min_error]["Rank"].values[0]
print("Minimum Rank:", min_rank)

df.to_csv("error_vs_rank.csv", index=False)



