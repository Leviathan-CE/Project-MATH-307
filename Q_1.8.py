from Q_1point6 import QRalgo
import numpy as np
A=np.array([[2,3]
        ,[-1,-2]]); A.shape=(2,2)
B=A+((0.9)*np.identity(2))
#shift
print("A:",A)
print("B:",B)
Bfinal=QRalgo(B,6)
#run Algo
print(Bfinal)
Bshiftback=Bfinal-((0.9)*np.identity(2))
#shift back
print(Bshiftback)
print(np.round(Bshiftback,1))
eigenvalues=np.diagonal(Bshiftback)
print("Eigenvalues using shift:",np.round(eigenvalues,1))
print("Using numpy function:",np.linalg.eigvals(A))

# Proof: 
# A*v=lamda*v, B=A+shift*I,(A+shift*I)*v=Av+shift*I*v
# = lamda*v+shift*v=(lamda+shift)*v, showing that lamda+ shift is
# an eigenvalue of B, therefore if lamda is an eigenvalue of A,
# then lamda + shift is an eigenvalue of B=A+shift*I



