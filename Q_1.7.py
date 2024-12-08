from Q_1point6 import QRalgo
import numpy as np

A=np.array([[2,3]
        ,[-1,-2]]); A.shape=(2,2)
QRalgo(A,6)
print("Using numpy function:",np.linalg.eigvals(A))

QRalgo(A,30)
print("Using numpy function:",np.linalg.eigvals(A))
# Can see the algorithm never converges on any eigvenvalues,
# looking at det(A-lamdaI)
# We can easily see A is not symmetric because
print(A,"A does not equal A^T: ",A.T)
# Therefore the matrix does not have real eigenvalues->
# it then has complex eigenvalues and therefore the QR
# algorithm can never converge to the eigenvalues.