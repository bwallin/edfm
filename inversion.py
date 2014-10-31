'''
Module containing formulations of the single pixel depth inversion problem
for the EDF microscopy project.

Author: Bruce Wallin
'''
from __future__ import division

from numpy import zeros, eye, dot, ones
from numpy.linalg import norm
import cvxopt
import cvxopt.solvers


class LinearFormulation():
    ''' This abstract class represents a linear formulation of the single
    pixel depth forward model and inversion problem.

    Attributes
    ----------
    A: array_like
        Model forward matrix (populated by set_psf_template method)
    y: array_like
        Observation vector
    x: array_like
        Solution vector (populated by solve method if successful)
    normalization_factor: float or array_like
        Factor(s) to scale system of equations for numerical stability.
    x_dim: int
        Length of unknown variables vector
    y_dim: int
        Length of observation vector

    '''
    def __init__(self, psf_template=None, image=None):
        self._psf_template = psf_template
        self._image = image
        self.A = None
        self.y = None

        self.normalization_factor = 1

        self.x_dim = None
        self.y_dim = None

    def set_psf_template(self):
        raise NotImplementedError

    def set_image(self):
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
        self.Q = None
        self.p = None
        self.G = None
        self.h = None

        self.result = None
        self.x = None

    def set_psf_template(self, psf_template):
        '''
        Read 3D array of discretized psf images and generate matrices A and Q.
        '''
        psf_template_tensor = psf_template.todense()
        r, p, q = psf_template_tensor.shape
        x_dim = r + 1
        A = zeros([p*q, r+1])  # +1 is for constant/intercept term
        for i in xrange(r):
            A[:, i] = psf_template_tensor[i, :, :].ravel()
        A[:, -1] = ones(p*q)  # This last column is for the constant term

        Q = 2*dot(A.transpose(), A)

        # self.normalization_factor = 1/norm(A, 2)
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
        A, y, Q, x_dim = self.A, self.y, self.Q, self.x_dim
        Q = cvxopt.matrix(Q)
        p = cvxopt.matrix(-2*dot(A.transpose(), y))
        G = cvxopt.matrix(-eye(x_dim))
        h = cvxopt.matrix(zeros((x_dim, 1)))
        result = cvxopt.solvers.qp(Q, p, G, h)

        self.result = result
        self.x = result['x']

        return result['status']

    def objective(self):
        return self.result['primal objective'] + (self.y.transpose(), self.y)


class LeastSquaresL1Reg(LinearFormulation):
    '''
    Formulation of minimization problem:

        min l2norm(Ax-y,2)^2 + alpha*l1norm(x)
        s.t. x >= 0
    '''
    def __init__(self, alpha=1):
        super(LeastSquaresL1Reg, self).__init__()

        self.alpha = 1

        self.Q = None
        self.p = None
        self.G = None
        self.h = None

    def set_psf_template(self, psf_template):
        '''
        Take PSF template and generate A and Q.
        '''
        p, q, r = psf_template.shape
        psf_template_tensor = psf_template.todense()
        A = zeros([p*q, r+1])  # +1 is for constant/intercept term
        for i in xrange(r):
            A[:, i] = psf_template_tensor[:, :, i].ravel()
        A[:, -1] = ones(p*q)  # This last column is for the constant term
        self.normalization_factor = 1/norm(A, 2)

        self.A = A
        Q = cvxopt.matrix(2*dot(A.transpose(), A))
        Z = cvxopt.spmatrix([], [], [], (r+1, r+1))
        self.Q = cvxopt.sparse([[Q, Z], [Z, Z]])

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
        p = cvxopt.matrix([cvxopt.matrix(-2*dot(A.transpose(), y)),
                           cvxopt.matrix(self.alpha*ones((r+1, 1)))])
        I = cvxopt.spdiag(cvxopt.matrix(ones((r+1, 1))))
        Z = cvxopt.spdiag(cvxopt.matrix(0*ones((r+1, 1))))
        G = cvxopt.sparse([[I, -I, -I], [-I, -I, Z]])
        z = cvxopt.matrix(0*ones((r+1, 1)))
        h = cvxopt.matrix([z, z, z])

        result = cvxopt.solvers.qp(Q, p, G, h)
        self.x = result['x']

        return result['x']


class L1NormEpigraph(LinearFormulation):
    '''
    Formulation of minimization problem:

        min norm(Ax - y, 1)
        s.t.       x >= 0
    '''
    def __init__(self):
        super(LinearFormulation, self).__init__()

        self.result = None

    def set_psf_template(self, psf_template):
        '''
        Read 3D array of discretized psf images and generate A.
        '''
        psf_template_tensor = psf_template.todense()
        r, p, q = psf_template_tensor.shape
        A = zeros([p*q, r+1])  # +1 for constant term
        for i in xrange(r):
            A[:, i] = psf_template_tensor[i, :, :].ravel()
        A[:, -1] = ones(p*q)  # This last column is for the constant term

        self.x_dim = r+1
        self.A = A

    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        self._image = image
        n, m = image.shape
        self.y_dim = n*m
        y = image.reshape((n*m, 1))*self.normalization_factor

        self.y = y

    def solve(self):
        '''
        "Solves" minimization problem using epigraph linear program
        formulation.
        '''
        A, y, x_dim, y_dim = self.A, self.y, self.x_dim, self.y_dim

        A = cvxopt.sparse(cvxopt.matrix(A))
        I_t = cvxopt.spdiag(cvxopt.matrix(ones(y_dim)))
        I_x = cvxopt.spdiag(cvxopt.matrix(ones(x_dim)))
        Z = cvxopt.spmatrix([], [], [], size=(x_dim, y_dim))
        A = cvxopt.sparse([
            [A, -A, -I_x],
            [-I_t, -I_t, Z]
            ])  # Sparse block matrix in COLUMN major order...for some reason

        y = cvxopt.matrix(y)
        z = cvxopt.matrix(zeros((x_dim, 1)))
        one = cvxopt.matrix(ones((y_dim, 1)))
        y = cvxopt.matrix([y, -y, z])
        c = cvxopt.matrix([z, one])

        result = cvxopt.solvers.lp(c, A, y, solver='glpk')

        self.result = result

        return result['status']

    def _solve_cvxopt_mosek(self):
        from mosek.fusion import Model, Domain, ObjectiveSense, Expr

        n, r = self.A
        with Model('L1 lp formulation') as model:
            model.variable('x', r, Domain.greaterThan(0.0))
            model.objective('l1 norm', ObjectiveSense.Minimize, Expr.dot())
            raise NotImplementedError
