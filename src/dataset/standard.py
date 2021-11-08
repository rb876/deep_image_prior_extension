from functools import partial
import odl
from odl.contrib.torch import OperatorModule
import torch
import numpy as np
from .ellipses import EllipsesDataset, DiskDistributedEllipsesDataset
from .brain import ACRINFMISOBrainDataset
from . import lotus
from . import walnuts
from util.matrix_ray_trafo import MatrixRayTrafo
from util.matrix_ray_trafo_torch import get_matrix_ray_trafo_module
from util.matrix_fbp_torch import get_matrix_fbp_module
from util.fbp import FBP


def subsample_angles_ray_trafo_matrix(matrix, cfg, proj_shape, order='C'):
    prod_im_shape = matrix.shape[1]

    matrix = matrix.reshape(
            (cfg.num_angles_orig, proj_shape[1] * prod_im_shape),
            order=order).tocsc()

    matrix = matrix[cfg.start:cfg.stop:cfg.step, :]

    matrix = matrix.reshape((np.prod(proj_shape), prod_im_shape),
                            order=order).tocsc()
    return matrix


def load_ray_trafo_matrix(name, cfg):

    if name in ['ellipses_lotus', 'ellipses_lotus_20',
                'ellipses_lotus_limited_45']:
        matrix = lotus.get_ray_trafo_matrix(cfg.ray_trafo_filename)
    # elif name == 'brain_walnut_120':  # currently useless as we can't use the
                                        # matrix impl for the walnut ray trafo,
                                        # because the filtering for FDK is not
                                        # implemented
    #     matrix = walnuts.get_ray_trafo_matrix(cfg.ray_trafo_filename)
    else:
        raise NotImplementedError

    return matrix


def get_ray_trafos(name, cfg, return_torch_module=True):
    """
    Return callables evaluating the ray transform and the smooth filtered
    back-projection for a standard dataset.

    The ray trafo can be implemented either by a matrix, which is loaded by
    calling :func:`load_ray_trafo_matrix`, or an odl `RayTransform` is used, in
    which case a standard cone-beam geometry is created.

    Subsampling of angles is supported for the matrix implementation only.

    Optionally, a ray transform torch module can be returned, too.

    Returns
    -------
    ray_trafos : dict
        Dictionary with the entries `'ray_trafo'`, `'smooth_pinv_ray_trafo'`,
        and optionally `'ray_trafo_module'`.
    """

    ray_trafos = {}

    if cfg.geometry_specs.impl == 'matrix':
        matrix = load_ray_trafo_matrix(name, cfg.geometry_specs)
        proj_shape = (cfg.geometry_specs.num_angles,
                      cfg.geometry_specs.num_det_pixels)
        if 'angles_subsampling' in cfg.geometry_specs:
            matrix = subsample_angles_ray_trafo_matrix(
                    matrix, cfg.geometry_specs.angles_subsampling, proj_shape)

        matrix_ray_trafo = MatrixRayTrafo(matrix,
                im_shape=(cfg.im_shape, cfg.im_shape),
                proj_shape=proj_shape)

        ray_trafo = matrix_ray_trafo.apply
        ray_trafos['ray_trafo'] = ray_trafo

        smooth_pinv_ray_trafo = FBP(
                matrix_ray_trafo.apply_adjoint, proj_shape,
                scaling_factor=cfg.fbp_scaling_factor,
                filter_type=cfg.fbp_filter_type,
                frequency_scaling=cfg.fbp_frequency_scaling).apply
        ray_trafos['smooth_pinv_ray_trafo'] = smooth_pinv_ray_trafo

        if return_torch_module:
            ray_trafos['ray_trafo_module'] = get_matrix_ray_trafo_module(
                    matrix, (cfg.im_shape, cfg.im_shape), proj_shape,
                    sparse=True)
            ray_trafos['smooth_pinv_ray_trafo_module'] = get_matrix_fbp_module(
                    get_matrix_ray_trafo_module(
                    matrix, (cfg.im_shape, cfg.im_shape), proj_shape,
                    sparse=True, adjoint=True), proj_shape,
                    scaling_factor=cfg.fbp_scaling_factor,
                    filter_type=cfg.fbp_filter_type,
                    frequency_scaling=cfg.fbp_frequency_scaling)

    elif cfg.geometry_specs.impl == 'custom':
        custom_cfg = cfg.geometry_specs.ray_trafo_custom
        if custom_cfg.name in ['walnut_single_slice',
                               'walnut_single_slice_matrix']:
            angles_subsampling = cfg.geometry_specs.angles_subsampling
            angular_sub_sampling = angles_subsampling.get('step', 1)
            # the walnuts module only supports choosing the step
            assert range(walnuts.MAX_NUM_ANGLES)[
                    angles_subsampling.get('start'):
                    angles_subsampling.get('stop'):
                    angles_subsampling.get('step')] == range(
                            0, walnuts.MAX_NUM_ANGLES, angular_sub_sampling)
            walnut_ray_trafo = walnuts.get_single_slice_ray_trafo(
                    data_path=custom_cfg.data_path,
                    walnut_id=custom_cfg.walnut_id,
                    orbit_id=custom_cfg.orbit_id,
                    angular_sub_sampling=angular_sub_sampling)
            if custom_cfg.name == 'walnut_single_slice':
                ray_trafos['ray_trafo'] = walnut_ray_trafo.apply
            elif custom_cfg.name == 'walnut_single_slice_matrix':
                matrix = walnuts.get_single_slice_ray_trafo_matrix(
                        path=custom_cfg.matrix_path,
                        walnut_id=custom_cfg.walnut_id,
                        orbit_id=custom_cfg.orbit_id,
                        angular_sub_sampling=angular_sub_sampling)
                matrix_ray_trafo = MatrixRayTrafo(matrix,
                        im_shape=(cfg.im_shape, cfg.im_shape),
                        proj_shape=(matrix.shape[0],))
                ray_trafos['ray_trafo'] = matrix_ray_trafo.apply

            # FIXME FDK is not smooth
            ray_trafos['smooth_pinv_ray_trafo'] = partial(
                    walnut_ray_trafo.apply_fdk, squeeze=True)

            if return_torch_module:
                if custom_cfg.name == 'walnut_single_slice':
                    ray_trafos['ray_trafo_module'] = (
                            walnuts.WalnutRayTrafoModule(walnut_ray_trafo))
                elif custom_cfg.name == 'walnut_single_slice_matrix':
                    ray_trafos['ray_trafo_module'] = (
                            get_matrix_ray_trafo_module(
                                    matrix, (cfg.im_shape, cfg.im_shape),
                                    (matrix.shape[0],), sparse=True))
                # ray_trafos['smooth_pinv_ray_trafo_module'] not implemented
        else:
            raise ValueError('Unknown custom ray trafo \'{}\''.format(
                    cfg.geometry_specs.ray_trafo_custom.name))

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
        if 'angles_subsampling' in cfg.geometry_specs:
            raise NotImplementedError

        ray_trafo = odl.tomo.RayTransform(space, geometry,
                impl=cfg.geometry_specs.impl)
        ray_trafos['ray_trafo'] = ray_trafo

        smooth_pinv_ray_trafo = odl.tomo.fbp_op(ray_trafo,
                filter_type=cfg.fbp_filter_type,
                frequency_scaling=cfg.fbp_frequency_scaling)
        ray_trafos['smooth_pinv_ray_trafo'] = smooth_pinv_ray_trafo

        if return_torch_module:
            ray_trafos['ray_trafo_module'] = OperatorModule(ray_trafo)
            ray_trafos['smooth_pinv_ray_trafo_module'] = OperatorModule(smooth_pinv_ray_trafo)

    return ray_trafos


def get_standard_dataset(name, cfg, return_ray_trafo_torch_module=True):
    """
    Return a standard dataset by name.
    """

    name = name.lower()

    ray_trafos = get_ray_trafos(name, cfg,
            return_torch_module=return_ray_trafo_torch_module)

    ray_trafo = ray_trafos['ray_trafo']
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

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
        space = lotus.get_domain128()
        proj_space = lotus.get_proj_space128()
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name == 'ellipses_lotus_20' or name == 'ellipses_lotus_limited_45':
        dataset_specs = {'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipses_dataset = EllipsesDataset(**dataset_specs)
        space = lotus.get_domain128()
        proj_space_orig = lotus.get_proj_space128()
        angles_coord_vector = proj_space_orig.grid.coord_vectors[0][
                cfg.geometry_specs.angles_subsampling.start:
                cfg.geometry_specs.angles_subsampling.stop:
                cfg.geometry_specs.angles_subsampling.step]
        proj_space = odl.uniform_discr_frompartition(
                odl.uniform_partition_fromgrid(
                        odl.RectGrid(angles_coord_vector,
                                     proj_space_orig.grid.coord_vectors[1])))
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name == 'brain_walnut_120':
        dataset_specs = {'data_path': cfg.data_path, 'shuffle': cfg.shuffle,
                         'zoom': cfg.zoom, 'zoom_fit': cfg.zoom_fit,
                         'random_rotation': cfg.random_rotation}
        brain_dataset = ACRINFMISOBrainDataset(**dataset_specs)
        space = brain_dataset.space
        proj_numel = cfg.geometry_specs.num_angles * cfg.geometry_specs.num_det_pixels
        proj_space = odl.rn(proj_numel, dtype=np.float32)
        dataset = brain_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    elif name == 'ellipses_walnut_120':
        dataset_specs = {'diameter': cfg.disk_diameter,
                         'image_size': cfg.im_shape, 'train_len': cfg.train_len,
                         'validation_len': cfg.validation_len, 'test_len': cfg.test_len}
        ellipses_dataset = DiskDistributedEllipsesDataset(**dataset_specs)
        space = ellipses_dataset.space
        proj_numel = cfg.geometry_specs.num_angles * cfg.geometry_specs.num_det_pixels
        proj_space = odl.rn(proj_numel, dtype=np.float32)
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=smooth_pinv_ray_trafo,
                domain=space, proj_space=proj_space,
                noise_type=cfg.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={'train': cfg.seed, 'validation': cfg.seed + 1,
                'test': cfg.seed + 2})
    else:
        raise NotImplementedError

    return dataset, ray_trafos


def get_test_data(name, cfg, return_torch_dataset=True):
    """
    Return external test data.

    E.g., for `'ellipses_lotus'` the scan of the lotus root is returned for
    evaluating a model trained on the `'ellipses_lotus'` standard dataset.

    Sinograms, FBPs and potentially ground truth images are returned, by
    default combined as a torch `TensorDataset` of two or three tensors.

    If `return_torch_dataset=False` is passed, numpy arrays
    ``sinogram_array, fbp_array, ground_truth_array`` are returned, where
    `ground_truth_array` can be `None` and all arrays have shape ``(N, W, H)``.
    """

    if cfg.test_data == 'lotus':
        sinogram, fbp, ground_truth = get_lotus_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]
        ground_truth_array = (ground_truth[None] if ground_truth is not None
                              else None)
    elif cfg.test_data == 'walnut':
        sinogram, fbp, ground_truth = get_walnut_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]  # FDK, actually
        ground_truth_array = ground_truth[None]
    else:
        raise NotImplementedError

    if return_torch_dataset:
        if ground_truth_array is not None:
            dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(sinogram_array[:, None]),
                        torch.from_numpy(fbp_array[:, None]),
                        torch.from_numpy(ground_truth_array[:, None]))
        else:
            dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(sinogram_array[:, None]),
                        torch.from_numpy(fbp_array[:, None]))

        return dataset
    else:
        return sinogram_array, fbp_array, ground_truth_array


def get_validation_data(name, cfg, return_torch_dataset=True):
    """
    Return external validation data.

    E.g., for `'ellipses_lotus'` data of the Shepp-Logan phantom is returned
    for validating a model trained on the `'ellipses_lotus'` standard dataset.

    Sinograms, FBPs and potentially ground truth images are returned, by
    default combined as a torch `TensorDataset` of two or three tensors.

    If `return_torch_dataset=False` is passed, numpy arrays
    ``sinogram_array, fbp_array, ground_truth_array`` are returned, where
    `ground_truth_array` can be `None` and all arrays have shape ``(N, W, H)``.
    """

    if cfg.validation_data == 'shepp_logan':
        sinogram, fbp, ground_truth = get_shepp_logan_data(name, cfg)
        sinogram_array = sinogram[None]
        fbp_array = fbp[None]
        ground_truth_array = (ground_truth[None] if ground_truth is not None
                              else None)
    else:
        raise NotImplementedError

    if return_torch_dataset:
        if ground_truth_array is not None:
            dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(sinogram_array[:, None]),
                        torch.from_numpy(fbp_array[:, None]),
                        torch.from_numpy(ground_truth_array[:, None]))
        else:
            dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(sinogram_array[:, None]),
                        torch.from_numpy(fbp_array[:, None]))

        return dataset
    else:
        return sinogram_array, fbp_array, ground_truth_array


def get_lotus_data(name, cfg):

    ray_trafos = get_ray_trafos(name, cfg,
                                return_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    sinogram = np.asarray(lotus.get_sinogram(
                    cfg.geometry_specs.ray_trafo_filename))
    if 'angles_subsampling' in cfg.geometry_specs:
        sinogram = sinogram[cfg.geometry_specs.angles_subsampling.start:
                            cfg.geometry_specs.angles_subsampling.stop:
                            cfg.geometry_specs.angles_subsampling.step, :]

    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    ground_truth = None
    if cfg.ground_truth_filename is not None:
        ground_truth = lotus.get_ground_truth(cfg.ground_truth_filename)

    return sinogram, fbp, ground_truth


def get_walnut_data(name, cfg):

    ray_trafos = get_ray_trafos(name, cfg,
                                return_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    angles_subsampling = cfg.geometry_specs.angles_subsampling
    angular_sub_sampling = angles_subsampling.get('step', 1)
    # the walnuts module only supports choosing the step
    assert range(walnuts.MAX_NUM_ANGLES)[
            angles_subsampling.get('start'):
            angles_subsampling.get('stop'):
            angles_subsampling.get('step')] == range(
                    0, walnuts.MAX_NUM_ANGLES, angular_sub_sampling)

    sinogram_full = walnuts.get_projection_data(
            data_path=cfg.data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id,
            angular_sub_sampling=angular_sub_sampling)

    # WalnutRayTrafo instance needed for selecting and masking the projections
    walnut_ray_trafo = walnuts.get_single_slice_ray_trafo(
            cfg.geometry_specs.ray_trafo_custom.data_path,
            walnut_id=cfg.geometry_specs.ray_trafo_custom.walnut_id,
            orbit_id=cfg.geometry_specs.ray_trafo_custom.orbit_id,
            angular_sub_sampling=angular_sub_sampling)

    sinogram = walnut_ray_trafo.flat_projs_in_mask(
            walnut_ray_trafo.projs_from_full(sinogram_full))

    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    slice_ind = walnuts.get_single_slice_ind(
            data_path=cfg.data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id)
    ground_truth = walnuts.get_ground_truth(
            data_path=cfg.data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id,
            slice_ind=slice_ind)

    return sinogram, fbp, ground_truth


def get_shepp_logan_data(name, cfg, modified=True, seed=30):

    dataset, ray_trafos = get_standard_dataset(
            name, cfg, return_ray_trafo_torch_module=False)
    smooth_pinv_ray_trafo = ray_trafos['smooth_pinv_ray_trafo']

    zoom = cfg.get('zoom', 1.)
    if zoom == 1.:
        ground_truth = odl.phantom.shepp_logan(dataset.space[1],
                                               modified=modified)
    else:
        full_shape = dataset.space[1].shape
        inner_shape = (int(zoom * dataset.space[1].shape[0]),
                       int(zoom * dataset.space[1].shape[1]))
        inner_space = odl.uniform_discr(
                min_pt=[-inner_shape[0] / 2, -inner_shape[1] / 2],
                max_pt=[inner_shape[0] / 2, inner_shape[1] / 2],
                shape=inner_shape)
        inner_ground_truth = odl.phantom.shepp_logan(inner_space,
                                                     modified=modified)
        ground_truth = dataset.space[1].zero()
        i0_start = (full_shape[0] - inner_shape[0]) // 2
        i1_start = (full_shape[1] - inner_shape[1]) // 2
        ground_truth[i0_start:i0_start+inner_shape[0],
                     i1_start:i1_start+inner_shape[1]] = inner_ground_truth
    ground_truth = (
            ground_truth /
            cfg.get('implicit_scaling_except_for_test_data', 1.)).asarray()

    random_gen = np.random.default_rng(seed)
    sinogram = dataset.ground_truth_to_obs(ground_truth, random_gen=random_gen)
    fbp = np.asarray(smooth_pinv_ray_trafo(sinogram))

    return sinogram, fbp, ground_truth
