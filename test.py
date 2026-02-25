import torch
import numpy as np
import math
import teacher

d = 7
p = 3
A = torch.randn(p, d, d)
A = 0.5 * (A + A.transpose(-1, -2))  # sym

bs = 1024
X = torch.randn(bs, d)
Z = teacher.flatten_H2_of_X_sym_hermite(X)               # (bs,m)

Aflat = teacher.flatten_A_sym_for_H2_feature(A)          # (p,m)
lhs = Z @ Aflat.T                                 # (bs,p)

# rhs
trA = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)         # (p,)
xAx = torch.einsum("bd,pde,be->bp", X, A, X)              # (bs,p)
rhs = (xAx - trA[None, :]) / math.sqrt(2.0)

print((lhs - rhs).abs().max().item())