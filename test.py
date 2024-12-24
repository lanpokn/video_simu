import numpy as np

def transform(x):
    # 第一步计算 1/2 + cos(1/3 * (arccos(2x - 1) + pi))
    intermediate = 1/2 + np.cos(1/3 * (np.arccos(2 * x - 1) + np.pi))
    return intermediate
print(transform(7/27))