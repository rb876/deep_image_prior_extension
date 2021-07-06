import torch

def random_brightness_contrast(images,
                               brightness_shift_range=(-0.2, 0.2),
                               contrast_factor_range=(0.8, 1.2),
                               clip_range=(0., None)):
    """
    Scale and shift the values of images randomly.

    Parameters
    ----------
    images : sequence of :class:`Tensor`
        Tensors to transform.
    brightness_shift_range : 2-tuple of float, optional
        Interval of the uniform distribution for sampling brightness shifts.
        The default is ``(-0.2, 0.2)``.
    contrast_factor_range : 2-tuple of float, optional
        Interval of the uniform distribution for sampling contrast factors.
        The default is ``(0.8, 1.2)``.
    clip_range : 2-tuple of (float or `None`) or `None`, optional
        Range into which the transformed image values are clipped.
        `None` disables clipping.
        The default is ``(0., None)``.

    Returns
    -------
    images : tuple
        Transformed tensors.
    """

    brightness_shift = (
        torch.rand(1) * (brightness_shift_range[1]-brightness_shift_range[0]) +
        brightness_shift_range[0])

    contrast_factor = (
        torch.rand(1) * (contrast_factor_range[1]-contrast_factor_range[0]) +
        contrast_factor_range[0])

    images_out = []

    for image in images:

        image_out = image * contrast_factor + brightness_shift

        if clip_range is not None:
            image_out = torch.clip(image_out,
                                   min=clip_range[0], max=clip_range[1])

        images_out.append(image_out)

    return tuple(images_out)
