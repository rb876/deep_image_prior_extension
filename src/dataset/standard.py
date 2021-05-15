import odl
from .ellipses import EllipsesDataset
from .lotus import get_ray_trafo_matrix, get_domain128, get_proj_space128
from util.matrix_ray_trafo import MatrixRayTrafo
from util.fbp import FBP

def get_standard_dataset(name, cfg):
    """
    Return a standard dataset by name.
    """

    name = name.lower()

    if cfg.geometry_specs.impl == 'matrix':
        matrix = get_ray_trafo_matrix(
                cfg.geometry_specs.ray_trafo_filename)
        matrix_ray_trafo = MatrixRayTrafo(matrix,
                im_shape=(cfg.im_shape, cfg.im_shape),
                proj_shape=(cfg.geometry_specs.num_angles,
                            cfg.geometry_specs.num_det_pixels))
        ray_trafo = matrix_ray_trafo.apply
        proj_shape = (cfg.geometry_specs.num_angles,
                      cfg.geometry_specs.num_det_pixels)
        smooth_pinv_ray_trafo = FBP(
                matrix_ray_trafo.apply_adjoint, proj_shape,
                scaling_factor=cfg.fbp_scaling_factor,
                filter_type=cfg.fbp_filter_type,
                frequency_scaling=cfg.fbp_frequency_scaling).apply
    else:
        space = odl.uniform_discr([-cfg.im_shape / 2, -cfg.im_shape / 2],
                                  [cfg.im_shape / 2, cfg.im_shape / 2],
                                  [cfg.im_shape, cfg.im_shape],
                                  dtype='float32')
        geometry = odl.tomo.cone_beam_geometry(space,
                src_radius=cfg.geometry_specs.src_radius,
                det_radius=cfg.geometry_specs.det_radius,
                num_angles=cfg.geometry_specs.num_angles,
                det_shape=cfg.geometry_specs.get('num_det_pixels', None))
        ray_trafo = odl.tomo.RayTransform(space, geometry,
                impl=cfg.geometry_specs.impl)
        proj_space = ray_trafo.range
        smooth_pinv_ray_trafo = odl.tomo.fbp_op(ray_trafo,
                filter_type=cfg.fbp_filter_type,
                frequency_scaling=cfg.fbp_frequency_scaling)

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
    elif name == 'ellipses_lotus':
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipses_dataset = EllipsesDataset(**dataset_specs)
        space = get_domain128()
        proj_space = get_proj_space128()
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    else:
        raise NotImplementedError

    return dataset, ray_trafo
