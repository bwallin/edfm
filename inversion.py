'''
Module containing formulations of the single pixel depth inversion problem
for the EDF microscopy project.

Author: Bruce Wallin
'''
from __future__ import division

from numpy import zeros, eye, dot, ones, floor
import cvxopt
import cvxopt.solvers

from misc import memoized


class Formulation(object):
    ''' This abstract class represents a formulation of the single
    pixel depth forward model and inversion problem.

    Attributes
    ----------
    x: array_like
        Solution vector (populated by solve method if successful)
    '''
    def __init__(self, psf_template=None, image=None):
        self._psf_template = psf_template
        self._image = image
        self.x = None

    def set_psf_template(self):
        '''
        Set the point-spread function template.
        '''
        raise NotImplementedError

    def set_image(self):
        '''
        Set the observed image.
        '''
        raise NotImplementedError

    def solve(self, pixels):
        '''
        Solve the inversion for the given pixel(s) in the observed image.
        '''
        raise NotImplementedError


class LeastSquaresCVXOPT(Formulation):
    '''
    This class represents a formulation of the single pixel depth inversion.
    Formulates and solves the following optimization problem:

        min l2norm(Ax-y)^2
        s.t.      x >= 0
    '''
    def __init__(self):
        super(LeastSquaresCVXOPT, self).__init__()
        self.psf_tensor = None
        self.image = None
        self.result = None

        self.A = None
        self.y = None
        self.x_dim = None
        self.y_dim = None
        self.normalization_factor = None

    def set_psf_template(self, psf_template):
        '''
        Read 3D array of discretized psf images and generate matrices A and Q.
        '''
        psf_tensor = psf_template.to_dense()
        # mask = reduce(lambda x, y: logical_or(x, y), psf_tensor)
        # bottom_proj = mask.sum(axis=0).astype('bool')
        # left_proj = mask.sum(axis=1).astype('bool')
        # min_i = list(bottom_proj).index(True)
        # max_i = list(bottom_proj).index(True, reversed=True)
        # min_j = list(left_proj).index(True)
        # max_j = list(left_proj).index(True, reversed=True)
        # psf_tensor = psf_tensor[min_i:max_i, min_j:max_j]

        self.psf_tensor = psf_tensor

    def set_image(self, image):
        '''
        Set image
        '''
        self.image = image

    # @memoized
    def _compute_A(self, bbox=(None, None, None, None)):
        lli, llj, uri, urj = bbox
        r, p, q = self.psf_tensor.shape
        x_dim, y_dim = r + 1, (uri-lli)*(urj-llj)  # +1 is for constant term
        A = zeros([y_dim, x_dim])
        for i in xrange(x_dim - 1):
            A[:, i] = self.psf_tensor[i, lli:uri, llj:urj].ravel()
        A[:, -1] = ones(y_dim)  # This last column is for the constant term

        self.x_dim, self.y_dim = x_dim, y_dim
        return A

    # @memoized
    def _compute_normalization(self, bbox=(None, None, None, None)):
        # normalization_factor = 1/norm(self.A, 2)
        normalization_factor = 1
        return normalization_factor

    @memoized
    def _compute_Q(self, bbox=(None, None, None, None)):
        Q = 2*dot(self.A.transpose(), self.A)
        return cvxopt.matrix(Q)

    def _compute_p(self):
        p = -2*dot(self.A.transpose(), self.y)
        return cvxopt.matrix(p)

    def _compute_G(self):
        G = -eye(self.x_dim)
        return cvxopt.matrix(G)

    def _compute_h(self):
        h = zeros((self.x_dim, 1))
        return cvxopt.matrix(h)

    def _compute_y(self, bbox=(None, None, None, None)):
        lli, llj, uri, urj = bbox
        clipped_image = self.image[lli:uri, llj:urj]
        n, m = clipped_image.shape
        y = clipped_image.reshape((n*m, 1))*self.normalization_factor
        return y

    def solve(self, pixel=None, solver='cvxopt.coneqp'):
        '''
        Solves minimization problem using quadratic program formulation.
        '''
        n, m = self.image.shape
        if not pixel:
            i, j = floor(n/2), floor(m/2)
        else:
            i, j = pixel
        r, p, q = self.psf_tensor.shape
        psf_bbox = (max(0, p/2-i), max(0, q/2-j),
                    min(p, p/2-i+n), min(q, q/2-j+m))
        img_bbox = (max(0, i-n/2), max(0, j-q/2),
                    min(n, i+n/2), min(m, j+m/2))
        self.A = self._compute_A(psf_bbox)
        self.normalization_factor = self._compute_normalization(psf_bbox)
        self.y = self._compute_y(img_bbox)

        Q = self._compute_Q()
        p = self._compute_p()
        G = self._compute_G()
        h = self._compute_h()
        result = cvxopt.solvers.qp(Q, p, G, h)

        self.result = result
        self.x = result['x']

        return result['status']

    def objective(self):
        return self.result['primal objective'] + (self.y.transpose(), self.y)


class LeastSquaresL1RegCVXOPT(Formulation):
    '''
    Formulation of minimization problem:

        min l2norm(Ax-y,2)^2 + alpha*l1norm(x)
        s.t. x >= 0
    '''
    def __init__(self, alpha=0):
        super(LeastSquaresL1RegCVXOPT, self).__init__()
        self.psf_tensor = None
        self.image = None
        self.result = None
        self.alpha = alpha

        self.A = None
        self.y = None
        self.x_dim = None
        self.y_dim = None
        self.normalization_factor = None

    def set_psf_template(self, psf_template):
        '''
        Take PSF template and generate A and Q.
        '''
        self.psf_tensor = psf_template.to_dense()

    def set_image(self, image):
        '''
        Take image and flatten into y.
        '''
        self.image = image

    def set_alpha(self, alpha):
        '''
        Set regularization parameter
        '''
        self.alpha = alpha

    def _compute_A(self, bbox=(None, None, None, None)):
        r, p, q = self.psf_tensor.shape
        x_dim, y_dim = r+1, p*q
        A = zeros([y_dim, x_dim])  # +1 is for constant/intercept term
        for i in xrange(x_dim-1):
            A[:, i] = self.psf_tensor[:, :, i].ravel()
        A[:, -1] = ones(y_dim)  # This last column is for the constant term
        self.x_dim, self.y_dim = x_dim, y_dim

        return A

    def _compute_normalization(self):
        # normalization_factor = 1/norm(self.A, 2)
        normalization_factor = 1
        return normalization_factor

    def _compute_y(self, bbox=(None, None, None, None)):
        n, m = self.image.shape
        y = self.image.reshape((n*m, 1))*self.normalization_factor
        return y

    def _compute_Q(self):
        Q = cvxopt.matrix(2*dot(self.A.transpose(), self.A))
        Z = cvxopt.spmatrix([], [], [], (self.x_dim, self.x_dim))
        Q = cvxopt.sparse([[Q, Z], [Z, Z]])
        return Q

    def _compute_p(self):
        p = cvxopt.matrix([cvxopt.matrix(-2*dot(self.A.transpose(), self.y)),
                           cvxopt.matrix(self.alpha*ones((self.x_dim, 1)))])
        return p

    def _compute_G(self):
        I = cvxopt.spdiag(cvxopt.matrix(ones((self.x_dim, 1))))
        Z = cvxopt.spdiag(cvxopt.matrix(0*ones((self.x_dim, 1))))
        G = cvxopt.sparse([[I, -I, -I], [-I, -I, Z]])
        return G

    def _compute_h(self):
        z = cvxopt.matrix(0*ones((self.x_dim, 1)))
        h = cvxopt.matrix([z, z, z])
        return h

    def solve(self):
        '''
        Solves minimization problem using quadratic program formulation.
        '''
        self.A = self._compute_A()
        self.normalization_factor = self._compute_normalization_factor()
        self.y = self._compute_y()
        Q = self._compute_Q()
        p = self._compute_p()
        G = self._compute_G()
        h = self._compute_h()
        result = cvxopt.solvers.qp(Q, p, G, h)
        self.x = result['x']

        return result['x']


class L1NormEpigraph(Formulation):
    '''
    Formulation of minimization problem:

        min norm(Ax - y, 1)
        s.t.       x >= 0
    '''
    def __init__(self):
        super(Formulation, self).__init__()

        self.result = None

    def set_psf_template(self, psf_template):
        '''
        Read 3D array of discretized psf images and generate A.
        '''
        psf_tensor = psf_template.to_dense()
        r, p, q = psf_tensor.shape
        A = zeros([p*q, r+1])  # +1 for constant term
        for i in xrange(r):
            A[:, i] = psf_tensor[i, :, :].ravel()
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
