import numpy as np

A=np.array([[2,3]
        ,[2,1]]); A.shape=(2,2)

B=np.array([[1,1]
        ,[2,1]]); B.shape=(2,2)

C=np.array([[1,0,-1]
        ,[1,2,1]
        ,[-4,0,1]]); C.shape=(3,3)

D=np.array([[1,1,-1]
        ,[0,2,0]
        ,[-2,4,2]]); D.shape=(3,3)

def QRalgo(A,numIterations):
    W=[]
    W.append(A)
    for i in range(1,numIterations):
        Q,R=np.linalg.qr(W[i-1],'complete')
        W.append(R@Q)
    print(np.round(W[1:numIterations],2))
    eigenvalues=np.diagonal(W[numIterations-1])
    print('Eigenvalues are:' ,np.round(eigenvalues,2))
    return W[numIterations-1]
    ## check rounding
print("A:\n")
QRalgo(A,6)
print("Using numpy function:",np.linalg.eigvals(A))
print("B:\n")
QRalgo(B,6)
print("Using numpy function:",np.linalg.eigvals(B))
print("C:\n")
QRalgo(C,6)
print("Using numpy function:",np.linalg.eigvals(C))
print("D:\n")
QRalgo(D,6)
print("Using numpy function:",np.linalg.eigvals(D))
# W=[]
# W.append(A)
# for i in range(1,6):
#     Q,R=np.linalg.qr(W[i-1],'complete')
#     W.append(R@Q)
# print(np.round(W[1:6],2))
# eigenvalues = np.diagonal(W[5])
# print('Eigenvalues are:' ,np.round(eigenvalues,2))
# ## check rounding





