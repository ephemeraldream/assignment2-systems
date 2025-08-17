import numpy as np
from einops import rearrange, reduce, einsum

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

xd = reduce(matrix, 'row column -> 1 row', 'max')
print(xd)