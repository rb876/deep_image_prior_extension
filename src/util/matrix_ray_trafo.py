import numpy as np


class MatrixRayTrafo:
    """
    Ray transform given by a (sparse) matrix.
    """
    def __init__(self, matrix, im_shape, proj_shape, order='C'):
        """
        Parameters
        ----------
        matrix : :class:`scipy.sparse.spmatrix` or array
            Matrix defining the mapping from images to projections.
            Must support matrix-vector multiplication via ``matrix.dot()``.
            Image and projection dimensions must be flattened in the specified
            `order`.
        im_shape : 2-tuple of int
            Image shape.
        proj_shape : 2-tuple of int
            Projection shape.
        order : {``'C'``, ``'F'``}, optional
            Order for reshaping images and projections from matrix shape to
            vector shape and vice versa.
            The default is ``'C'``.
        """
        self.matrix = matrix
        self.im_shape = im_shape
        self.proj_shape = proj_shape
        self.order = order

    def apply(self, x):
        """
        Apply the forward projection by (sparse) matrix multiplication.
        """
        x_flat = np.reshape(np.asarray(x), -1, order=self.order)
        y_flat = self.matrix.dot(x_flat)
        y = np.reshape(y_flat, self.proj_shape, order=self.order)
        return y

    def apply_adjoint(self, y):
        """
        Apply the adjoint by (sparse) matrix multiplication.
        """
        y_flat = np.reshape(np.asarray(y), -1, order=self.order)
        x_flat = self.matrix.T.dot(y_flat)
        x = np.reshape(x_flat, self.im_shape, order=self.order)
        return x
