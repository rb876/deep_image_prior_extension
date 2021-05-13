import odl
from .ellipses import EllipsesDataset

def get_standard_dataset(name, cfg):
    """
    Return a standard dataset by name.
    """

    name = name.lower()

    space = odl.uniform_discr([-cfg.im_shape / 2, -cfg.im_shape / 2],
                              [cfg.im_shape / 2, cfg.im_shape / 2],
                              [cfg.im_shape, cfg.im_shape],
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
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipses_dataset = EllipsesDataset(**dataset_specs)
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo, noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    else:
        raise NotImplementedError

    return dataset, ray_trafo
