# cuda_sprec

A sparse signal recovery library written in PyCUDA.

## Overview
Sparse recovery algorithms attempt to solve problems in which you have an undetermined system of linear equations, but there the solution is known to be sparse. This can be formalized as follows:

Let $A$ be an $m \times n$ (with $m < n$)  matrix and $b \in \mathbb{R}^m$ be a vector. Find a vector $x \in \mathbb{R}^n$ such that $Ax=b$ and $x$ has the smallest possible number of non-zero entries (`l0` minimization problem, NP-hard).
- $A$ is the measurement matrix that models the transformation of a high-dimensional signal to an observed lower-dimensional signal, $b$.
- $x$ is the original high-dimensional and sparse signal to recover.
    - This may be a high-dimensional signal where only a few frequency bands are activate, and it's usually represented by a vector where most entries are 0.
    - $x = [0,0,3,0,0,-2,0,0,0,1,0,0,0,0,0]$ is 15-dimensional, and only 3 frequencies are active.

The measurement matrix $A$ can be referred to as a "dictionary". A dictionary is a set of "atoms" or basis vectors (columns of $A$) from which signals can be reconstructed. If $x$ is sparse, this means the signal can be approximately reconstructed from a small number of these atoms.

## Greedy Algorithms
These algorithms iteratively select the dictionary (measurement matrix)  element that best correlates with the current residual, i.e., the difference between the observed vector $b$ and the vector formed by multiplying $A$ and the current estimate of $x$. The residual at any iteration $i$ is defined as $r=b-Ax^{(i)}$.

### Orthogonal Matching Pursuit (OMP)

At ech iteration, select the column of $A$ most correlated with the current residual, add it to the support of the solution, an dadjust the current solution to be the least squares solution consistent with the current support.
1. Initialize $r=b,S=[]$ (support)
2. Iterate until the correlated with the current residual, add it to the support of the solution, an dadjust the current solution to be the least squares solution consistent with the current support.
3. Iterate until a stopping condition is met:
    - $i=\text{argmax}_j\|<r,A_j>\|$
    - $S=S\cup \{i\}$
    - $x_S=\text{argmin}_x\|b-A_Sx\|_2$
    - $r=b-A_Sx_S$
