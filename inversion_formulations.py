'''
Main library for positivity constrained least squares formulation of depth inversion
for EDF microscopy project.

Author: Bruce Wallin
'''
from __future__ import division
import logging
import pdb

from numpy import zeros, eye, dot, ones, vstack
from numpy.linalg import norm
import cvxopt
import cvxopt.solvers
from cvxopt import solvers, sparse, matrix, spdiag, spmatrix
import tifffile


class single_loc_lsq_cvxopt():
    '''
    Formulation of minimization problem:
    min norm(Ax-y)^2
    s.t.      x >= 0
    '''
    def __init__(self):
        self.A = None
        self.y = None
        self.normalization_factor = None

        # cvxopt ready arrays
        self.Q = None
        self.p = None
        self.G = None
        self.h = None


    def set_psf_template(self, psf_template_tensor):
        '''
        Read 3D array of discretized PSF images and generate A and Q.
        '''
        p,q,r = psf_template_tensor.shape
        A = zeros([p*q, r+1]) # +1 is for constant/intercept term
        for i in xrange(r):
            A[:,i] = psf_template_tensor[:,:,i].ravel()
        A[:, -1] = ones(p*q) # This last column is for the constant/intercept term

        self.A = A
        Q = 2*dot(A.transpose(), A)
        self.Q = matrix(Q)


    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        self.y = image.ravel()


    def solve(self):
        '''
        Solves minimization problem using quadratic program formulation.
        '''
        A, y, Q = self.A, self.y, self.Q
        n, n = Q.size
        p = matrix(-2*dot(A.transpose(), y))
        G = matrix(-eye(n))
        h = matrix(zeros((n,1)))
        result = solvers.qp(Q, p, G, h)

        return result['x']


class single_loc_lsql1reg_cvxopt():
    '''
    Formulation of minimization problem:
    min norm(Ax-y,2)^2 + alpha*norm(x, 1)
    s.t. x >= 0
    '''
    def __init__(self):
        self.A = None
        self.y = None
        self.alpha = None
        self.normalization_factor = None

        # cvxopt ready arrays
        self.Q = None
        self.p = None
        self.G = None
        self.h = None


    def set_psf_template(self, psf_template_tensor):
        '''
        Read 3D array of discretized PSF images and generate A and Q.
        '''
        p,q,r = psf_template_tensor.shape
        A = zeros([p*q, r+1]) # +1 is for constant/intercept term
        for i in xrange(r):
            A[:,i] = psf_template_tensor[:,:,i].ravel()
        A[:, -1] = ones(p*q) # This last column is for the constant/intercept term
        self.normalization_factor = 1/norm(A)

        self.A = A
        Q = matrix(2*dot(A.transpose(), A))
        Z = spmatrix([], [], [], (r+1, r+1))
        self.Q = sparse([[Q, Z],[Z, Z]])


    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        n, m = image.shape
        self.y = self.normalization_factor*image.ravel().reshape([n*m, 1])


    def set_alpha(self, alpha):
        self.alpha = alpha


    def solve(self):
        '''
        Solves minimization problem using quadratic program formulation.
        '''
        A, y, Q = self.A, self.y, self.Q
        n, n = Q.size
        r = n/2-1
        p = matrix([matrix(-2*dot(A.transpose(), y)), matrix(self.alpha*ones((r+1,1)))])
        I = spdiag(matrix(ones((r+1, 1))))
        Z = spdiag(matrix(0*ones((r+1, 1))))
        G = sparse([[I, -I, -I], [-I, -I, Z]])
        z = matrix(0*ones((r+1, 1)))
        h = matrix([z, z, z])
        result = solvers.qp(Q, p, G, h)

        return result['x']


class multi_loc_lsq_cvxopt():
    '''
    Formulation of minimization problem:
    min norm(Ax-y)^2
    s.t.      x >= 0
    '''
    def __init__(self):
        self.A = None
        self.y = None

        # cvxopt ready arrays
        self.Q = None
        self.p = None
        self.G = None
        self.h = None


    def set_num_locs(num_locs):
        self.num_locs = num_locs


    def set_psf_template(self, psf_template_tensor):
        '''
        Read 3D array of discretized PSF images and generate A and Q.
        '''
        n = self.num_locs
        p,q,r = psf_template_tensor.shape
        A = zeros([p*q, r*n+1]) # +1 is for constant/intercept term
        for i in xrange(r*n):
            A[:,i] = psf_template_tensor[:,:,i].ravel()
        A[:, -1] = ones(p*q) # This last column is for the constant/intercept term

        self.A = A
        Q = 2*dot(A.transpose(), A)
        self.Q = matrix(Q)


    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        self.y = image.ravel()


    def solve(self):
        '''
        Solves minimization problem using quadratic program formulation.
        '''
        A, y, Q = self.A, self.y, self.Q
        n, n = Q.size
        try:
            p = matrix(-2*dot(A.transpose(), y))
        except:
            pdb.set_trace()
        G = matrix(-eye(n))
        h = matrix(zeros((n,1)))
        result = solvers.qp(Q, p, G, h)

        return result['x']



class lsq_formulation_mosek():
    '''
    Formulation of minimization problem:
    min norm(Ax-y)^2
    s.t.      x >= 0

    Using mosek solver
    '''
    def __init__(self):
        self.A = None
        self.y = None

        # cvxopt ready arrays
        self.Q = None
        self.p = None
        self.G = None
        self.h = None


    def set_psf_template(self, psf_template_tensor):
        '''
        Read 3D array of discretized PSF images and generate A and Q.
        '''
        r,p,q = psf_template_tensor.shape
        A = zeros([p*q, r+1]) # +1 is for constant/intercept term
        for i in xrange(r):
            A[:,i] = psf_template_tensor[i,:,:].ravel()
        A[:, -1] = ones(p*q) # This last column is for the constant/intercept term

        self.A = A
        self.Q = 2*dot(A.transpose(), A)


    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        self.y = image.ravel()


    def solve(self):
        '''
        Solves minimization problem using quadratic program formulation.
        '''
        A, y, Q = self.A, self.y, self.Q
        n, n = Q.shape
        p = matrix(-2*dot(A.transpose(), y))
        G = matrix(-eye(n))
        h = matrix(zeros((n,1)))
        result = solvers.qp(Q, p, G, h)

        return result['x']


    def solve(self):
        import mosek
        import mosek.fusion
        from mosek.fusion import Model, Domain, ObjectiveSense, Expr

        n, r = self.A
        with Model('L1 lp formulation') as model:
            model.variable('x', r, Domain.greaterThan(0.0))
            model.objective('l1 norm', ObjectiveSense.Minimize, Expr.dot())



class l1_formulation_cvxopt():
    '''
    Formulation of minimization problem:
    min l1norm(Ax - y)
    s.t.       x >= 0
    '''
    def __init__(self):
        self.A = None
        self.y = None
        self.c = None


    def set_psf_template(self, psf_template_tensor):
        '''
        Read 3D array of discretized PSF images and generate A.
        '''
        p,q,r = psf_template_tensor.shape
        A = zeros([p*q, r+1]) # +1 for constant/intercept term
        for i in xrange(r):
            A[:,i] = psf_template_tensor[:,:,i].ravel()
        A[:, -1] = ones(p*q) # This last column is for the constant/intercept term

        A = sparse(matrix(A))
        I_pq = spdiag(matrix(ones(p*q)))
        I_r = spdiag(matrix(ones(r+1)))
        Z = spmatrix([], [], [], size=(r+1, p*q))
        A = sparse([
            [A, -A, -I_r],
            [-I_pq, -I_pq, Z]
            ]) # Sparse block matrix in COLUMN major order...for some reason
        c = vstack([
            zeros((r+1, 1)),
            ones((p*q, 1))
            ])

        self.c = matrix(c)
        self.A = A


    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        y = image.ravel()
        self.y = y


    def solve(self):
        '''
        Solves minimization problem using epigraph linear program formulation.
        '''
        A, y, c = self.A, self.y, self.c
        N,M = A.size
        n = len(y)
        y = matrix(vstack([
            y.reshape((n,1)),
            -y.reshape((n,1)),
            zeros((N-2*n,1))]))
        result = solvers.lp(c, A, y)

        return result['x']
