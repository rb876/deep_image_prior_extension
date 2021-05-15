import numpy as np
import odl
from odl.discr import ResizingOperator
from odl.trafos import FourierTransform


class FBP:
    """
    Filtered back-projection applying a filter returned by
    :func:`get_fbp_filter_op`, followed by calling a specified adjoint.
    """
    def __init__(self, adjoint_func, *args, **kwargs):
        """
        Parameters
        ----------
        adjoint_func : callable
            Callable calculating the adjoint of the ray transform.
            Receives a projection and returns an image.
        *args, **kwargs
            Arguments passed to :func:`get_fbp_filter_op`.
        """
        self.adjoint_func = adjoint_func
        self.filter_op = get_fbp_filter_op(*args, **kwargs)

    def apply(self, y):
        """
        Apply the filtered back-projection.
        """
        y = self.filter_op(y)
        x = self.adjoint_func(y)
        return x


# copy of odl.tomo.analytic.filtered_back_projection._fbp_filter
def _fbp_filter(norm_freq, filter_type, frequency_scaling):
    """Create a smoothing filter for FBP.

    This function is a copy of https://github.com/odlgroup/odl/blob/25ec783954a85c2294ad5b76414f8c7c3cd2785d/odl/tomo/analytic/filtered_back_projection.py#L49
    in order to not rely on potential implementation changes.
    All rights remain with the ODL authors.

    Parameters
    ----------
    norm_freq : `array-like`
        Frequencies normalized to lie in the interval [0, 1].
    filter_type : {'Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann',
                   callable}
        The type of filter to be used.
        If a string is given, use one of the standard filters with that name.
        A callable should take an array of values in [0, 1] and return the
        filter for these frequencies.
    frequency_scaling : float
        Scaling of the frequencies for the filter. All frequencies are scaled
        by this number, any relative frequency above ``frequency_scaling`` is
        set to 0.

    Returns
    -------
    smoothing_filter : `numpy.ndarray`

    Examples
    --------
    Create an FBP filter

    >>> norm_freq = np.linspace(0, 1, 10)
    >>> filt = _fbp_filter(norm_freq,
    ...                    filter_type='Hann',
    ...                    frequency_scaling=0.8)
    """
    filter_type, filter_type_in = str(filter_type).lower(), filter_type
    if callable(filter_type):
        filt = filter_type(norm_freq)
    elif filter_type == 'ram-lak':
        filt = np.copy(norm_freq)
    elif filter_type == 'shepp-logan':
        filt = norm_freq * np.sinc(norm_freq / (2 * frequency_scaling))
    elif filter_type == 'cosine':
        filt = norm_freq * np.cos(norm_freq * np.pi / (2 * frequency_scaling))
    elif filter_type == 'hamming':
        filt = norm_freq * (
            0.54 + 0.46 * np.cos(norm_freq * np.pi / (frequency_scaling)))
    elif filter_type == 'hann':
        filt = norm_freq * (
            np.cos(norm_freq * np.pi / (2 * frequency_scaling)) ** 2)
    else:
        raise ValueError('unknown `filter_type` ({})'
                         ''.format(filter_type_in))

    indicator = (norm_freq <= frequency_scaling)
    filt *= indicator
    return filt


def get_fbp_filter_op(proj_space, scaling_factor=1., padding=True,
                      filter_type='Ram-Lak', frequency_scaling=1.0):
    """
    Clone of :func:`odl.tomo.fbp_filter_op` for 2D geometries that does not
    rely on an :class:`odl.tomo.RayTransform` object.
    No scaling other than the user-specified `scaling_factor` is applied.

    This function is an edited copy of https://github.com/odlgroup/odl/blob/25ec783954a85c2294ad5b76414f8c7c3cd2785d/odl/tomo/analytic/filtered_back_projection.py#L313 .
    Rights remain with the ODL authors.

    Parameters
    ----------
    proj_space : :class:`odl.DiscretizedSpace` or 2-sequence of int
        Projection space, with dimension 0 specifying angles and dimension 1
        specifying the position on the detector.
        If a tuple is specified, the space
        ``odl.uniform_discr([0., -0.5], [2.*np.pi, 0.5], proj_space)`` is used.
        Note that the choice of `min_pt` and `max_pt` is not relevant for the
        resulting array values.
    scaling_factor : float, optional
        Constant scaling factor. In contrast to :func:`odl.tomo.fbp_filter_op`,
        no scaling is applied by default.

    Other Parameters
    ----------------
    See documentation of :func:`odl.tomo.fbp_filter_op`.

    Returns
    -------
    filter_op : :class:`odl.Operator`
        Filter operator for a filtered back-projection (FBP), close to the
        implementation in ODL.
    """
    if not isinstance(proj_space, odl.DiscretizedSpace):
        # Construct proj_space from shape specification
        # `min_pt` and `max_pt` are not relevant for the resulting values
        proj_space = odl.uniform_discr(
                [0., -0.5], [2.*np.pi, 0.5], proj_space, dtype='float32')

    # Define ramp filter
    def fourier_filter(x):
        abs_freq = np.abs(x[1])
        norm_freq = abs_freq / np.max(abs_freq)
        filt = _fbp_filter(norm_freq, filter_type, frequency_scaling)
        return filt
        # scaling = 1 / (2 * alen)
        # return filt * np.max(abs_freq) * scaling

    # Define (padded) fourier transform
    if padding:
        # Define padding operator
        ran_shp = (proj_space.shape[0],
                   proj_space.shape[1] * 2 - 1)
        resizing = ResizingOperator(proj_space, ran_shp=ran_shp)

        fourier = FourierTransform(resizing.range, axes=1)
        fourier = fourier * resizing
    else:
        fourier = FourierTransform(proj_space, axes=1)

    # Create ramp in the detector direction
    ramp_function = fourier.range.element(fourier_filter)
    
    # scaling_factor = 1
    # scaling_factor *= domain.cell_volume
    # scaling_factor /= proj_space.cell_volume

    ramp_function *= scaling_factor

    # Create ramp filter via the convolution formula with fourier transforms
    return fourier.inverse * ramp_function * fourier
