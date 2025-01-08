""" API Call
import numpy as np
X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)

# Expected Output:
# [[1.660477 2.660477]
#  [2.339523 3.339523]]
"""

import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def check_input(Q, K, V):
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    assert Q.shape[1] == K.shape[1]
    return K.shape[1]

def softmax(m:np.array):
    max_el = np.max(m, axis=1, keepdims=True)
    m = m - max_el
    exp_m = np.exp(m)
    sum_exp = np.sum(exp_m, axis=1, keepdims=True)
    return exp_m/sum_exp

def self_attention(Q, K, V):
    dv = check_input(Q, K , V)
    mtx = (1.0/np.sqrt(dv)) * Q @ K.T
    mtx = softmax(mtx)
    return mtx @ V

def test1():
    X = np.array([[1, 0], [0, 1]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    output = self_attention(Q, K, V)
    print(np.round(output,5))
    assert np.alltrue(np.round(output,5) == np.array([ [1.66048, 2.66048],  [2.33952, 3.33952]]))

if __name__=="__main__":
    test1()