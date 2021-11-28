import tensorly as tl
from sympy import *
from tensorly.decomposition import parafac
from tensorly.decomposition import non_negative_parafac_hals

def check_rank(tensor, rank, non_neg=True, n=10, tol=0.001, p=False):
    if non_neg:
        for k in range(n):
            weights, factors = non_negative_parafac_hals(tensor, n_iter_max=1000000, rank=rank, init='random')
            full = tl.cp_to_tensor((weights, factors))
            diff = (full - tensor) / tensor
            if p:
                print(tl.max(abs(diff)))
            if tl.max(abs(diff)) < tol:
                return True
    else:
        for k in range(n):
            weights, factors = parafac(tensor, n_iter_max=1000000, rank=rank)
            full = tl.cp_to_tensor((weights, factors))
            diff = (full - tensor) / tensor
            if p:
                print(tl.max(abs(diff)))
            if tl.max(abs(diff)) < tol:
                return True
    return False


# determines sign of a list
def sgn(a):
    t = 0
    ab = 0
    for a_i in a:
        t+= abs(a_i)
        ab += a_i
    return t == abs(ab)

# checks if the given 2x2x3 has all nonneg rank 3 sub 2x2x2
def check(t):
    t1 = tl.tensor([t[0], t[1]])
    t2 = tl.tensor([t[1], t[2]])
    t3 = tl.tensor([t[0], t[2]])
    a1 = det(Matrix(t1[0]))
    a2 = det(Matrix(t1[1]))
    a3 = det(Matrix(t2[0]))
    a4 = det(Matrix(t2[1]))
    b1 = det(Matrix(t1[:,0]))
    b2 = det(Matrix(t1[:,1]))
    b3 = det(Matrix(t2[:,0]))
    b4 = det(Matrix(t2[:,1]))
    c1 = det(Matrix(t1[:,:,0]))
    c2 = det(Matrix(t1[:,:,1]))
    c3 = det(Matrix(t2[:,:,0]))
    c4 = det(Matrix(t2[:,:,1]))

    a5 = det(Matrix(t3[0]))
    a6 = det(Matrix(t3[1]))

    b5 = det(Matrix(t3[:,0]))
    b6 = det(Matrix(t3[:,1]))

    c5 = det(Matrix(t3[:,:,0]))
    c6 = det(Matrix(t3[:,:,1]))
    return sgn([a1, a2,a3,a4,a5,a6]) or sgn([b1,b2,b3,b4,b5,b6]) or sgn([c1,c2,c3,c4,c5,c6])


# checks if the give 2x2x2 tensor has nonneg rank 3
def check_r3(t):
    a1 = det(Matrix(t[0]))
    a2 = det(Matrix(t[1]))
    b1 = det(Matrix(t[:,0]))
    b2 = det(Matrix(t[:,1]))
    c1 = det(Matrix(t[:,:,0]))
    c2 = det(Matrix(t[:,:,1]))
    return sgn([a1,a2]) or sgn([b1,b2]) or sgn([c1,c2])

# checks if given 2x2x2 has nonneg rank 2
def check_r2(t):
    a1 = det(Matrix(t[0]))
    a2 = det(Matrix(t[1]))
    b1 = det(Matrix(t[:,0]))
    b2 = det(Matrix(t[:,1]))
    c1 = det(Matrix(t[:,:,0]))
    c2 = det(Matrix(t[:,:,1]))
    d1 = ineq(a1,a2,b1,b2,c1,c2,ge,ge,ge)
    d2 = ineq(a1,a2,b1,b2,c1,c2,le,le,ge)
    d3 = ineq(a1,a2,b1,b2,c1,c2,le,ge,le)
    d4 = ineq(a1,a2,b1,b2,c1,c2,ge,le,le)
    supermod = d1 or d2 or d3 or d4
    return supermod


# helpers
def ineq(a1,a2,b1,b2,c1,c2,f1,f2,f3):
    t1 = f1(a1,0) and f1(a2,0)
    t2 = f2(b1,0) and f2(b2,0)
    t3 = f3(c1,0) and f3(c2,0)
    return t1 and t2 and t3
    
def ge(a1,a2):
    return a1 >= a2

def le(a1,a2):
    return a1 <= a2

# Checks if 2x2x3 contains a 2x2x2 with nonneg 4
def check_r4(t, upper = 1):
    a1 = not check_r3(tl.tensor([t[0],t[1]]))
    a2 = not check_r3(tl.tensor([t[0],t[2]]))
    a3 = not check_r3(tl.tensor([t[1],t[2]]))
    return (a1 + a2 + a3) >= upper

def r2_sub(t, upper = 1):
    a = check_r2(tl.tensor([t[0],t[1]]))
    b = check_r2(tl.tensor([t[0],t[2]]))
    c = check_r2(tl.tensor([t[2],t[1]]))
    return (bool(a)+bool(b)+bool(c)) >= upper


def mat_inv(t):
    tens = t.copy()
    M_1 = Matrix([Matrix(tens[0][1]).transpose(),Matrix(tens[0][0]).transpose()])
    M_2 = Matrix([Matrix(tens[1][1]).transpose(),Matrix(tens[1][0]).transpose()])
    M_3 = Matrix([Matrix(tens[2][1]).transpose(),Matrix(tens[2][0]).transpose()])
    tens[0] = M_1
    tens[1] = M_2
    tens[2] = M_3
    return tens

def mat_trans(t):
    tens = t.copy()
    tens[0] = tens[0].transpose()
    tens[1] = tens[1].transpose()
    tens[2] = tens[2].transpose()
    return tens

def rotate(t):
    return mat_inv(mat_trans(t))
