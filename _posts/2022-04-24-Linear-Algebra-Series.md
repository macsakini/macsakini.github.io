---
toc : true
---
Linear Algebra Series 1 - Introduction

This particular blog post will cover at large the basics of linear algebra and how to understand it in thirty minutes. Cap. I would give me a year😬.

We are gonna use a lot of Python for the computations so be ready to do some nice copy paste. The first section involves importing all necessary libraries into the workspace. In this case they are only two:🥲. Our star library for today is the __SciPy__ library which will mostly deal with scientific calculations and as such scientific subjects, disciples, I mean disciplines, notations, et al....

```python
import numpy as np
import scipy.linalg as sp
```
Follow by, creating a matrix using the array function by numpy. The formula remains standard 😊 __RC which is Roman Catholic which means Rows then Columns__

```python
a = np.array([[1,2,3],[4,5,6],[7,88,9]])
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
```
## Basic Operations on Matrices

**Finding the determinant.** NOTE:= The determinant is a scalar value. 
Say we have a matrix A = [[a,b],[c,d]]:
We find the detriminant using the formula $$ Det(A) = (a*d)-(c*b) $$. The determinant and __this is not a formal definition__, can be thought of as the "volume" of the space occupied by the matrix.

```python
sp.det(a)
```
If the determinant is zero, this matrix becomes a __singular matrix__. Otherwise, it is just a regular square matrix. I say square as there exist other types of Matrices, but these will be discussed in other sections of the series.

__B__ is an example singular matrix. Ensure to test with B as well, in place of A, just to get a feel of the concept.

Next, we can find the norm of the matrix := |A|. Similar to any vector, matrices have a norm which is just their magnitude (size).

The norm of the matrix can be derived using the norm function.

```python
sp.norm(a)
```

The inverse of the matrix. As the word implies, is the alternative matrix in the opposite, say what, "direction" but orthogonally. Putting it in another way, interchange a and d, give b and c, negative signs and "vwaalah" inverse.
[[a,b],[c,d]] \\ \Longrightarrow \\ [[d,-b], [-c,a]]. However, a more formal way of doing it is using the inv function provided by SciPy. Mathematicians, use a matrix of signs to find the inverse for bigger matrices. 3 x 3, 4 x 4 and the gang. 

Note a singular matrix, does not have an inverse.
```python
try:
  print(sp.inv(a))
except:
  print("This matrix is a singular matrix")
```

## Finding the Eigenvalues and Eigenvectors
This section just shows the way of finding the eigen's using SciPy. However, I will add a link down below to an exclusive post, fully focused, on what these are, how the help and where to use them. They carry a major role in CS, Physics, Chemistry... as such they are very important.

Finding the eigenvalues and eigenvectors.
```python
sp.eig(a)
```
This specifically prints out the eigenvalues.
```python
sp.eigvals(a)
```
## Lets talk about matrix decomposition
There are so many ways of decomposing a matrix. Name a few, LU, QR, QZ, Polar, SVD and so much more. You are probably wondering, who are all these Chinese babies. However, i will disappoint you, these are decomposition styles. I will cover only the most important among these. 

They all try to achieve the same task: make matrix computations easier. Think of decomposition as making matrices into __legos__. The matrix doesn't loose its end value, we just break it into pieces, that when we remap together reform the broken down matrix. This happens so that they can be used in Ninjago and be used differently and more efficiently at various tasks. Imagine, 9 is a matrix, if I break it into 4 and 5 or even 2, 4 and 3, we havent really changed the 9. I have just broken it down or __decomposed__ it into a number of values. Decomposition is somewhat similar, not direct, yes, but somewhat similar.

### LU Decomposition
This is a pivoted LU decomposition.
```python
lu = sp.lu(a)

print(lu[0], "\n")
print(lu[1], "\n")
print(lu[2], "\n")
```

```python
sp.lu_factor(a)
```
### Singular Value Decomposition
This is an SVD.
```python
svd = sp.svd(a)

print(svd[0], "\n")
print(svd[1], "\n")
print(svd[2], "\n")
```
```python
sp.svdvals(a)
```
#### Find the orthonormal basis for A using SVD
Basis and orthonormal shall be covered in the next post. Be sure to head there.
```python
sp.orth(a)
```
#### Finds the orthonormal basis for the null_space using SVD
```python
sp.null_space(a)
```
### Polar Decomposition
```python
sp.polar(a)
```
## Matrix can also be considered as regular numbers with regular applications.
This definitively means, you can carry out operations similar to those on real and natural numbers without any siginificant changes. A number of the operations include:

1. Finding the exponential of the matrix.
```python
sp.expm(a)
```
2. Finding the sine of the matrix.
```python
sp.sinm(a)
```
3. Finding the cosine.
```python
sp.cosm(a)
```
4. Finding the tangent of the matrix.
```python
sp.tanm(a)
```
5. Finding the square root of the matrix.
```python
sp.sqrtm(a)
```
6. And even applying a function to the matrix. For example, this is the dot product of the matrix.
```python
print(sp.funm(a, lambda x: x*x))

print("\n")

print(np.dot(a,a))
```

## There is a whole lot of special matrices. However i will just build a few to demonstrate.

This creates a 3 by 3 hilbert matrix. Hilbert spaces have special characteristics coupled up with the Inner Product and the Banach Space and so on... but i will also discuss this in later sections and in full length. I mean, this is the introductory, you wouldn't want me on Banach Spaces, right.
```python
sp.hilbert(3)
```
Hadamard matrix is also a special matrix. Some arab guy made it. Its mostly used in quantum computations due to its nature of just being 1's and -1's, big deal, whoop. ITS A VERY BIG DEAL.🫡
.They are used for the opening and closing of logic gates in quantum computers coupled with superposition which i will not even dare mention.
```python
sp.hadamard(4) 
```
This creates a discrete fourier transform matrix. FO-RII-EEEH. 
```python
sp.dft(3) 
```
This creates a helmert matrix. Its kind of used with the hadamard, but do some research.
```python
sp.helmert(3) 
```

That mostly sums it up. This is an introductory session on using the scipy library to evaluate some of Linear Algebra's most common functions. In the next, sections, I will get in deeper into the aforementioned topics, be there. Stay Tuned.😌