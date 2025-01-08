"""
Write a Python function to implement the Positional Encoding layer for Transformers.
The function should calculate positional encodings for a sequence length (position) and
    model dimensionality (d_model) using sine and cosine functions as specified in the
    Transformer architecture.
The function should return -1 if position is 0, or if d_model is less than or equal to 0.
The output should be a numpy array of type float16.
"""
import numpy as np

def pos_encoding(seq_len: int, d_model: int):
    """
    pe[p,2i]   = sin(p/base^(2i/d_model))
    pe[p,2i+1] = cos(p/base^((2i+1)/d_model))
    :param seq_len:
    :param d_model:
    :return:
    """
    base = 10_000
    half = (d_model+1)//2
    if d_model<1 or seq_len<1:
        return -1
    freq = np.exp(-np.log(base) * 2 * np.arange(half)/d_model)[None,:]
    pos = np.arange(seq_len)[:,None]
    sin = np.sin(pos * freq)
    cos = np.cos(pos * freq)
    if d_model % 2 == 1:
        cos = cos[:,:-1]

    pos_enc = np.zeros((seq_len,d_model))
    pos_enc[:,0::2] = sin
    pos_enc[:,1::2] = cos
    pos_enc = np.float16(pos_enc)
    return pos_enc

if __name__=="__main__":
    pe = pos_encoding(4, 9)
    print(pe.shape)
    print(pe)