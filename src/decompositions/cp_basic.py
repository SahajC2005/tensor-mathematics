import numpy as np

X = np.zeros((2,2,2))

X[:,:,0] = np.array([[1,2],
                     [3,4]])
X[:,:,1] = np.array([[2,1],
                     [1,3]])

print("Original tensor:\n", X)
a = np.array([1.0, 1.0])
b = np.array([1.0, 1.0])
c = np.array([1.0, 1.0])

print("b =", b)
print("c =", c)
print("-"*40)


def update_a(X, b, c):
    num = np.zeros(2)
    for i in range(2):
        s = 0
        for j in range(2):
            for k in range(2):
                s += X[i,j,k] * b[j] * c[k]
        num[i] = s
    denom = np.dot(b,b) * np.dot(c,c)
    return num / denom

def update_b(X, a, c):
    num = np.zeros(2)
    for j in range(2):
        s = 0
        for i in range(2):
            for k in range(2):
                s += X[i,j,k] * a[i] * c[k]
        num[j] = s
    denom = np.dot(a,a) * np.dot(c,c)
    return num / denom

def update_c(X, a, b):
    num = np.zeros(2)
    for k in range(2):
        s = 0
        for i in range(2):
            for j in range(2):
                s += X[i,j,k] * a[i] * b[j]
        num[k] = s
    denom = np.dot(a,a) * np.dot(b,b)
    return num / denom

a_new = update_a(X, b, c)
b_new = update_b(X, a_new, c)
c_new = update_c(X, a_new, b_new)

print("Updated factors:")
print("a =", a_new)
print("b =", b_new)
print("c =", c_new)
print("-"*40)