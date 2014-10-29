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
import tifffile


class LinearFormulation(Object):
    ''' This abstract class represents a linear formulation of the single
    pixel depth inversion.

    Attributes
    ----------
    A: array_like
        Model forward matrix (populated by set_PSF_template method)
    y: array_like
        Observation vector
    x: array_like
        Solution vector (populated by solve method if successful)
    normalization_factor: float
        Factor to scale system of equations for numerical stability.
    x_dim: int
        Length of unknown variables vector
    y_dim: int
        Length of observation vector

    '''
    def __init__(self, psf_template, image):
        self._psf_template = psf_template
        self._image = image
        self.A = None # Forward model matrix
        self.y = None # Flattened observed image

        self.normalization_factor = None # Normalization factor for numerical convenience

        self.x_dim = None # Number of unknown variables
        self.y_dim = None # Number of pixels in observed image


    @property
    def psf_template(self):
        raise NotImplementedError


    @property
    def image(self):
        raise NotImplementedError


    def solve(self):
        raise NotImplementedError


class LeastSquares(LinearFormulation):
    '''
    This class represents a formulation of the single pixel depth inversion.
    Formulates and solves the following optimization problem:

    min l2norm(Ax-y)^2
    s.t.      x >= 0
    '''
    def __init__(self):
        super(LeastSquares, self).__init__()
        self.Q = None # Quadratic second order term
        self.p = None # Quadratic linear term
        self.G = None # Constraint matrix
        self.h = None # Constraint lower bounds

        self.result = None # Result of 'cvxopt.solvers.qp', populated after call to 'solve'


    def set_PSF_template(self, PSF_template):
        '''
        Read 3D array of discretized PSF images and generate matrices A and Q.
        '''
        r,p,q = PSF_template_tensor.shape
        x_dim = r+1
        A = zeros([p*q, r+1]) # +1 is for constant/intercept term
        for i in xrange(r):
            A[:,i] = PSF_template_tensor[i,:,:].ravel()
        A[:, -1] = ones(p*q) # This last column is for the constant/intercept term

        #normalization_factor = 1/norm(A, 2)
        normalization_factor = 1

        Q = 2*dot(A.transpose(), A)

        self.normalization_factor = normalization_factor
        self.x_dim = x_dim
        self.A = A
        self.Q = Q


    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        n, m = image.shape
        y = image.reshape((n*m, 1))*self.normalization_factor

        self.y = y
        self.y_dim = len(y)


    def solve(self, solver='cvxopt.coneqp'):
        '''
        Solves minimization problem using quadratic program formulation.
        '''
        pass


    def _solve_cvxopt_coneqp(self):
        A, y, Q, x_dim = self.A, self.y, self.Q, self.x_dim
        Q = cvxopt.matrix(Q)
        p = cvxopt.matrix(-2*dot(A.transpose(), y))
        G = cvxopt.matrix(-eye(x_dim))
        h = cvxopt.matrix(zeros((x_dim,1)))
        result = cvxopt.solvers.qp(Q, p, G, h)

        self.result = result
        self.x = result['x']

        return result['status']



    def objective(self):
        return result['primal objective'] + (self.y.transpose(), self.y)


class LeastSquaresL1Reg(LinearFormulation):
    '''
    Formulation of minimization problem:
    min l2norm(Ax-y,2)^2 + alpha*l1norm(x)
    s.t. x >= 0
    '''
    def __init__(self):
        super(LeastSquaresL1Reg, self).__init__()

        self.Q = None
        self.p = None
        self.G = None
        self.h = None


    def set_PSF_template(self, PSF_template):
        '''
        Read 3D array of discretized PSF images and generate A and Q.
        '''
        p,q,r = PSF_template_tensor.shape
        A = zeros([p*q, r+1]) # +1 is for constant/intercept term
        for i in xrange(r):
            A[:,i] = PSF_template_tensor[:,:,i].ravel()
        A[:, -1] = ones(p*q) # This last column is for the constant/intercept term
        self.normalization_factor = 1/norm(A)

        self.A = A
        Q = cvxopt.matrix(2*dot(A.transpose(), A))
        Z = spmatrix([], [], [], (r+1, r+1))
        self.Q = cvxopt.sparse([[Q, Z],[Z, Z]])


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
        p = cvxopt.matrix([cvxopt.matrix(-2*dot(A.transpose(), y)), cvxopt.matrix(self.alpha*ones((r+1,1)))])
        I = spdiag(cvxopt.matrix(ones((r+1, 1))))
        Z = spdiag(cvxopt.matrix(0*ones((r+1, 1))))
        G = cvxopt.sparse([[I, -I, -I], [-I, -I, Z]])
        z = cvxopt.matrix(0*ones((r+1, 1)))
        h = cvxopt.matrix([z, z, z])

        result = cvxopt.solvers.qp(Q, p, G, h)
        self.x = result['x']

        return result['x']


class LeastSquaresMosek():
    '''
    Formulation of minimization problem:
    min norm(Ax-y)^2
    s.t.      x >= 0

    Using mosek solver
    '''
    def __init__(self):
        super(LinearFormulation, self).__init__()

        # cvxopt ready arrays
        self.Q = None
        self.p = None
        self.G = None
        self.h = None


    def set_PSF_template(self, PSF_template_tensor):
        '''
        Read 3D array of discretized PSF images and generate A and Q.
        '''
        r,p,q = PSF_template_tensor.shape
        A = zeros([p*q, r+1]) # +1 is for constant/intercept term
        for i in xrange(r):
            A[:,i] = PSF_template_tensor[i,:,:].ravel()
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
        p = cvxopt.matrix(-2*dot(A.transpose(), y))
        G = cvxopt.matrix(-eye(n))
        h = cvxopt.matrix(zeros((n,1)))
        result = cvxopt.solvers.qp(Q, p, G, h)

        return result['x']


    def solve(self):
        import mosek
        import mosek.fusion
        from mosek.fusion import Model, Domain, ObjectiveSense, Expr

        n, r = self.A
        with Model('L1 lp formulation') as model:
            model.variable('x', r, Domain.greaterThan(0.0))
            model.objective('l1 norm', ObjectiveSense.Minimize, Expr.dot())



class L1NormCvxoptGLPK(LinearFormulation):
    '''
    Formulation of minimization problem:
    min norm(Ax - y, 1)
    s.t.       x >= 0
    '''
    def __init__(self):
        super(LinearFormulation, self).__init__()

        self.result = None


    def set_PSF_template(self, PSF_template_tensor):
        '''
        Read 3D array of discretized PSF images and generate A.
        '''
        r,p,q = PSF_template_tensor.shape
        A = zeros([p*q, r+1]) # +1 for constant/intercept term
        for i in xrange(r):
            A[:,i] = PSF_template_tensor[i,:,:].ravel()
        A[:, -1] = ones(p*q) # This last column is for the constant/intercept term

        self.normalization_factor = 1 # Normalization factor for numerical convenience

        self.x_dim = r+1
        self.A = A


    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        n, m = image.shape
        self.y_dim = n*m
        y = image.reshape((n*m, 1))*self.normalization_factor

        self.y = y


    def solve(self):
        '''
        "Solves" minimization problem using epigraph linear program formulation.
        '''
        A, y, x_dim, y_dim  = self.A, self.y, self.x_dim, self.y_dim

        A = cvxopt.sparse(cvxopt.matrix(A))
        I_t = spdiag(cvxopt.matrix(ones(y_dim)))
        I_x = spdiag(cvxopt.matrix(ones(x_dim)))
        Z = spmatrix([], [], [], size=(x_dim, y_dim))
        A = cvxopt.sparse([
            [A, -A, -I_x],
            [-I_t, -I_t, Z]
            ]) # Sparse block matrix in COLUMN major order...for some reason

        y = cvxopt.matrix(y)
        z = cvxopt.matrix(zeros((x_dim,1)))
        one = cvxopt.matrix(ones((y_dim, 1)))
        y = cvxopt.matrix([y, -y, z])
        c = cvxopt.matrix([z, one])

        result = cvxopt.solvers.lp(c, A, y, solver='glpk')

        self.result = result

        return result['status']
