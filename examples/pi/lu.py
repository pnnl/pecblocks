# copyright 2022-2023 Battelle Memorial Institute
import numpy as np

A=np.array([[1,-2],[-2,5]], dtype=float)
B=np.array([1,0], dtype=float)

A=np.array([[5,7,6,5],
            [7,10,8,7],
            [6,8,10,9],
            [5,7,9,10]], dtype=float)

B=np.array([1,0,2,-1], dtype=float)

# direct solution of Aq=B
q=np.linalg.solve(A,B)
print ('A', A)
print ('B', B)
print ('Direct q', q)
print ('Check', np.matmul(A, q))

L=np.linalg.cholesky(A)
#U=L.T
print ('\nL', L)
# print ('U', U)

# forward substations; Ly = B
n = len(B)
y = B
for i in range(n):
  for j in range(i):
    y[i] -= L[i,j] * y[j]
  y[i] /= L[i,i]

# back substitutions; L'q = y
q = np.zeros(n)
for i in range(n,0,-1):
  q[i-1] = (y[i-1] - np.dot(L[i:,i-1],q[i:])) / L[i-1,i-1]

print ('Cholesky q', q)
print ('Check', np.matmul(A, q))

