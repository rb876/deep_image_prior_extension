import odl
from .ellipses import EllipsesDataset

def get_standard_dataset(name, cfg):
    """
    Return a standard dataset by name.
    """

    name = name.lower()

    space = odl.uniform_discr([-cfg.dataset_specs.im_shape[0] / 2, -cfg.dataset_specs.im_shape[0] / 2],
                              [cfg.dataset_specs.im_shape[0] / 2, cfg.dataset_specs.im_shape[0] / 2],
                              [cfg.dataset_specs.im_shape[0], cfg.dataset_specs.im_shape[1]],
                              dtype='float32')
    geometry = odl.tomo.cone_beam_geometry(space,
            src_radius=cfg.geometry_specs.src_radius,
            det_radius=cfg.geometry_specs.det_radius,
            num_angles=cfg.geometry_specs.num_angles)
    ray_trafo = odl.tomo.RayTransform(space, geometry,
            impl=cfg.geometry_specs.impl)
    smooth_pinv_ray_trafo = odl.tomo.fbp_op(ray_trafo,
            filter_type='Hann', frequency_scaling=0.6)

    if cfg.noise_specs.noise_type == 'white':
        specs_kwargs = {'stddev': cfg.noise_specs.stddev}
    elif cfg.noise_specs.noise_type == 'poisson':
        specs_kwargs = {'mu_water': cfg.noise_specs.mu_water,
                        'photons_per_pixel': cfg.noise_specs.photons_per_pixel
                        }
    else:
        raise NotImplementedError

    if name == 'ellipses':
        ellipses_dataset = EllipsesDataset(image_size=cfg.dataset_specs.im_shape[0])
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo, noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.dataset_specs.seed, 'validation': cfg.dataset_specs.seed + 1,
                'test': cfg.dataset_specs.seed + 2})
    else:
        raise NotImplementedError

    return dataset, ray_trafo
