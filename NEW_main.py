import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

# this is restarted b/c I noticed we only need black and white
# and color was making it very complicated, using lots of code from old main

A=mpimg.imread('unnamed.jpg')

A = np.dot(A[...,:3], [0.2989, 0.5870, 0.1140])
#Convert to greyscale 
print(A,"\n")
print(A.shape,"\n")

#Print np arrays as images
plt.imshow(A, interpolation='nearest',cmap='gray')
plt.show()


U, S, VT = np.linalg.svd(A)
print("U matrix:\n", U)
print("Singular values:\n", S)
print("V^T matrix:\n", VT)

# Rank 1 A1 below

# part c

UVT=np.outer(U[:,0],VT[0,:])
print("UVT:",UVT)
A1r=S[0]*UVT

plt.imshow(A1r, interpolation='nearest',cmap='gray')
plt.show()

# Part d

# For A2

UVT2=np.outer(U[:,1],VT[1,:])
A2=S[1]*UVT2

plt.imshow(A2, interpolation='nearest',cmap='gray')
plt.show()

Ak=np.add(A1r,A2)

plt.imshow(Ak, interpolation='nearest',cmap='gray')
plt.show()

Ai=np.zeros(A.shape)
for i in range(0,50):
    ## matrice addition
    NewAddition=S[i]*(np.outer(U[:,i],VT[i,:]))
    Ai=Ai+NewAddition
# 50 is a reasonable image

Ak=Ai
# e

fig = plt.figure()

# First subplot for image A
ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
ax1.imshow(A, interpolation='nearest',cmap='gray')
ax1.set_title('Image A')

# Second subplot for image Ak
ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
ax2.imshow(Ak, interpolation='nearest',cmap='gray')
ax2.set_title('Image Ak (50)')

# Show the figure with both images
plt.show()

Aw=np.zeros(A.shape)
for i in range(0,75):
    ## matrice addition
    NewAddition=S[i]*(np.outer(U[:,i],VT[i,:]))
    Ai=Ai+NewAddition
#TODO experiment how many make an acceptable image

Aw=Ai

fig = plt.figure()

# First subplot for image A
ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
ax1.imshow(A, interpolation='nearest',cmap='gray')
ax1.set_title('Image A')

# Second subplot for image Ak
ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
ax2.imshow(Aw, interpolation='nearest',cmap='gray')
ax2.set_title('Image Aw (75)')

# Show the figure with both images
plt.show()
#but 75 looks better