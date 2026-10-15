# Code Explanation: Tensor Decomposition and Polynomial Regression

## Overview
This code performs tensor decomposition on video frames and finds the relationship between decomposition rank and reconstruction error using polynomial regression.

---

## Part 1: Imports and Data Loading (Lines 2-16)

```python
import cv2, cp, numpy, tensorly, sklearn, matplotlib
frame = cp.frame
video = cp.video
video_tensor = cp.video_tensor
```

**What it does:**
- Imports libraries for video processing, tensor operations, and machine learning
- Loads preprocessed video data from `cp.py`:
  - `frame`: List of grayscale frames (64x64 pixels each)
  - `video`: Video capture object
  - `video_tensor`: 3D numpy array (64×64×N frames) - the video as a tensor

---

## Part 2: Class Definitions

### Class 1: `decompose` (Lines 18-33)
**Purpose:** Process video frames (not actively used in main code)

**How it works:**
- Reads video frame by frame
- Resizes each frame to 64×64 pixels
- Converts to grayscale
- Stacks frames into a 3D tensor

---

### Class 2: `rank` (Lines 35-55)
**Purpose:** Performs CP (Canonical Polyadic) tensor decomposition

**Key components:**
- `__init__`: Stores the video tensor
- `parfac()`: Performs PARAFAC decomposition with a given rank
  - **Rank**: Number of components to extract (lower rank = more compression)
  - Uses Alternating Least Squares (ALS) algorithm
  - Returns factors A, B, C that can reconstruct the original tensor

**What is tensor decomposition?**
- Breaks down a 3D tensor into simpler components
- Like PCA but for 3D data
- Original tensor ≈ A × B × C (with rank components)

---

### Class 3: `Error` (Lines 58-78)
**Purpose:** Calculates reconstruction error for different ranks

**How it works:**
1. `__init__()`:
   - Takes video tensor and a rank value
   - Performs decomposition at that rank
   - Stores the factors (A, B, C)

2. `calculate()`:
   - Reconstructs the tensor from factors: `Y = reconstruct(A, B, C)`
   - Computes error: `||original - reconstructed||`
   - Computes relative error: `error / ||original||`
   - Stores result in `Error_dict[rank] = relative_error`

**Error_dict**: Class variable that accumulates all rank→error pairs

---

## Part 3: Main Loop - Testing Different Ranks (Lines 80-83)

```python
for i,j in enumerate(range(1, 64)):
    error = Error(video_tensor, j, i)
    error.calculate()
```

**What it does:**
- Tests ranks from 1 to 63
- For each rank:
  - Creates an `Error` object
  - Calculates reconstruction error
  - Stores it in `Error.Error_dict`

**Result:** Dictionary mapping: `{rank_1: error_1, rank_2: error_2, ...}`

---

## Part 4: Visualization (Lines 86-90)

```python
plt.plot(Error.Error_dict.keys(), Error.Error_dict.values())
plt.show()
```

**What it shows:**
- Line plot: Rank (x-axis) vs Error (y-axis)
- Shows how error decreases as rank increases
- Higher rank = better reconstruction = lower error

---

## Part 5: Polynomial Regression Analysis (Lines 93-117)

### Step 1: Prepare Data (Lines 93-100)
```python
error_values = list of errors (y-axis values)
rank_values = list of ranks (x-axis values)
sort_idx = indices to sort by error values
x = sorted error values  # Input feature
y = sorted rank values   # Target to predict
```

**Why sort?**
- Polynomial features work better with sorted data
- Ensures smooth curve fitting

### Step 2: Test Different Polynomial Degrees (Lines 102-114)
```python
for d in degrees (1 to 9):
    1. Create polynomial features of degree d
       - Degree 1: [x, 1] → linear
       - Degree 2: [x², x, 1] → quadratic
       - Degree 3: [x³, x², x, 1] → cubic
       - etc.
    
    2. Fit linear regression on polynomial features
    
    3. Evaluate using 5-fold cross-validation:
       - Split data into 5 parts
       - Train on 4 parts, test on 1 part
       - Repeat 5 times, average the MSE
       - More reliable than single train/test split
    
    4. Store the average MSE
```

**Cross-validation explained:**
- Prevents overfitting to a specific train/test split
- Gives more reliable error estimates
- 5-fold = data split 5 ways, tested 5 times

### Step 3: Find Best Degree (Lines 116-117)
```python
print("MSE for each degree:", errors)
print("Best degree:", degrees[np.argmin(errors)])
```

**What it does:**
- Prints MSE for each polynomial degree
- Finds degree with lowest MSE (best fit)
- This tells you the relationship shape:
  - Degree 1 = linear relationship
  - Degree 2 = quadratic (U-shaped curve)
  - Degree 3+ = more complex curves

---

## Summary: What the Code Does

1. **Decomposes video** using different ranks (1-63)
2. **Measures error** for each rank
3. **Finds relationship** between error and rank using polynomial regression
4. **Determines best polynomial degree** to model this relationship

**Goal:** Understand how reconstruction quality (error) relates to decomposition complexity (rank), so you can choose the optimal rank for your needs.

---

## Key Concepts

- **Rank**: Number of components in decomposition (higher = better quality, more storage)
- **Error**: How different reconstructed tensor is from original (lower = better)
- **Polynomial Regression**: Fits curves (not just lines) to data
- **Cross-validation**: Robust way to evaluate models without overfitting


