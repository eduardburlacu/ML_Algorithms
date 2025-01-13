import numpy as np

def convolve(mtx, kernel):
    assert len(mtx.shape)==3
    h, w, c = mtx.shape
    dx,dy = kernel.shape
    mtx_pad = np.pad(
        mtx,
        ((dx//2,dx//2),(dy//2,dy//2),(0,0)),
        mode="constant"
    )
    out=np.zeros_like(mtx)
    for i in range(h):
        for j in range(w):
            out[i,j,:] = np.sum(
                kernel[:,:,None] * mtx_pad[i:i + dx, j:j+dy, :],
                axis=(0, 1)
            )
    return out
eps = 1.0
mtx = eps * np.random.rand(10,10,1) + np.ones((10,10,3))
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
grad_x = convolve(mtx, sobel_x)
grad_y = convolve(mtx, sobel_y)
grad_magn = np.sqrt(grad_x**2 + grad_y**2)
bits = 8
min_val =0
max_val = 2**bits - 1
min_mtx = np.min(grad_magn)
max_mtx = np.max(grad_magn)
grad_magn = ((grad_magn-min_mtx)/(max_mtx-min_mtx) * (max_val-min_val) + min_val).astype(np.uint8)
print(grad_magn.transpose(2, 0, 1))
