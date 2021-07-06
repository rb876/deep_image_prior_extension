import numpy as np
try:
    from elasticdeform import deform_grid
    ELASTICDEFORM_AVAILABLE = True
except ImportError:
    ELASTICDEFORM_AVAILABLE = False


def deform_random_grid(X, sigma=25, points=3, order=3, mode='constant',
                       cval=0.0, crop=None, prefilter=True, axis=None,
                       affine=None, rotate=None, zoom=None, random_gen=None):
    """
    Elastic deformation with a random deformation grid

    This function is copied and adapted from Gijs van Tulder's
    ``elasticdeform`` repository https://github.com/gvtulder/elasticdeform/blob/14471c47c474c806fdfde541ba6457c8471e959e/elasticdeform/deform_grid.py#L6
    and also uses the ``elasticdeform`` package.

    This generates a random, square deformation grid with displacements
    sampled from from a normal distribution with standard deviation `sigma`.
    The deformation is then applied to the image or list of images,

    See ``deform_grid`` for a full description of the parameters.

    Parameters
    ----------
    X : numpy array or list of arrays
        image, or list of images of the same size
    sigma : float
        standard deviation of the normal distribution
    points : array
        number of points of the deformation grid
    rotate : float or None
        angle in degrees to rotate the output

        This only works for 2D images and rotates the image around
        the center of the output.
    zoom : float or None
        scale factor to zoom the output

        This only works for 2D images and scales the image around
        the center of the output.

    See Also
    --------
    deform_grid : for a full description of the parameters.
    """
    assert ELASTICDEFORM_AVAILABLE, 'missing the `elasticdeform` module'
    # prepare inputs and axis selection
    Xs = _deform_grid_normalize_inputs(X)
    axis, deform_shape = _deform_grid_normalize_axis_list(axis, Xs)

    if random_gen is None:
        random_gen = np.random.default_rng()

    if not isinstance(points, (list, tuple)):
        points = [points] * len(deform_shape)

    displacement = random_gen.randn(len(deform_shape), *points) * sigma
    return deform_grid(X, displacement, order, mode, cval, crop, prefilter, axis, affine, rotate, zoom)


# taken from https://github.com/gvtulder/elasticdeform/blob/14471c47c474c806fdfde541ba6457c8471e959e/elasticdeform/deform_grid.py#L295
def _deform_grid_normalize_inputs(X):
    if isinstance(X, np.ndarray):
        Xs = [X]
    elif isinstance(X, list):
        Xs = X
    else:
        raise Exception('X should be a numpy.ndarray or a list of numpy.ndarrays.')

    # check X inputs
    assert len(Xs) > 0, 'You must provide at least one image.'
    assert all(isinstance(x, np.ndarray) for x in Xs), 'All elements of X should be numpy.ndarrays.'
    return Xs


# taken from https://github.com/gvtulder/elasticdeform/blob/14471c47c474c806fdfde541ba6457c8471e959e/elasticdeform/deform_grid.py#L308
def _deform_grid_normalize_axis_list(axis, Xs):
    if axis is None:
        axis = [tuple(range(x.ndim)) for x in Xs]
    elif isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = [axis] * len(Xs)
    assert len(axis) == len(Xs), 'Number of axis tuples should match number of inputs.'
    input_shapes = []
    for x, ax in zip(Xs, axis):
        assert isinstance(ax, tuple), 'axis should be given as a tuple'
        assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
        assert len(ax) == len(axis[0]), 'All axis tuples should have the same length.'
        assert ax == tuple(set(ax)), 'axis must be sorted and unique'
        assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
        input_shapes.append(tuple(x.shape[d] for d in ax))
    assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
    deform_shape = input_shapes[0]
    return axis, deform_shape
