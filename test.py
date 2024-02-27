import numpy as np

P = [
    [13, 12, -2],
    [12, 17, 6],
    [-2 , 6, 12]

]

x = np.array([1, 0.5, -1]).T

q = [-22, -14.5, 13]
print(np.matmul(P,x) + q)