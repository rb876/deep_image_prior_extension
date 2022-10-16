"""
Utilities for displaying results (e.g. consistent titles).
"""

experiment_title_dict = {
    'pretrain': 'EDIP (noise)',  # 'EDIP (switch to noise)'
    'pretrain_only_fbp': 'EDIP (FBP)',
    'pretrain_noise': 'EDIP (FBP & noise)',
    'no_pretrain': 'DIP (noise)',
    'no_pretrain_fbp': 'DIP (FBP)',
    'no_pretrain_2inputs': 'DIP (FBP & noise)',
}

data_title_dict = {
    'ellipses_lotus_20': 'Ellipses-Lotus (Sparse 20)',
    'ellipses_lotus_limited_45': 'Ellipses-Lotus (Limited 45)',
    'brain_walnut_120': 'Brain-Walnut (Sparse 120)',
    'ellipses_walnut_120': 'Ellipses-Walnut (Sparse 120)',
    'ellipsoids_walnut_3d': 'Ellipsoids-Walnut (3D Sparse 20)',
    'ellipsoids_walnut_3d_60': 'Ellipsoids-Walnut (3D Sparse 60)',
    'meta_pretraining_lotus_20': 'MAML Lotus (Sparse 20)',
}

experiment_color_dict = {
    'pretrain_only_fbp': '#A42A2E',
    'pretrain': '#663399',
    'no_pretrain': '#000000',
    'no_pretrain_fbp': '#3D78B2',
}

def get_title_from_run_spec(run_spec):
    experiment_title = run_spec.get('experiment_title')
    if experiment_title is None:
        experiment_title = experiment_title_dict.get(run_spec['experiment'],
                                                     run_spec['experiment'])

    name_title = run_spec.get('name_title')
    if name_title is None:
        name_title = run_spec.get('name') or ''

    title_prefix = run_spec.get('title_prefix', '')
    title_suffix = run_spec.get('title_suffix', '')

    title = ''.join([
        title_prefix,
        experiment_title,
        ' [{}]'.format(name_title) if name_title else '',
        title_suffix,
        ])

    return title

def get_data_title_full(data, validation_run):
    if validation_run:
        part = {
                'ellipses_lotus_20': 'Shepp-Logan',
                'ellipses_lotus_limited_45': 'Shepp-Logan',
                'brain_walnut_120': 'Shepp-Logan',
                'ellipses_walnut_120': 'Shepp-Logan',
                'ellipsoids_walnut_3d': 'Shepp-Logan',
                'ellipsoids_walnut_3d_60': 'Shepp-Logan',
                }[data]
        data_title_part = 'Validation on {}'.format(part)
    else:
        part = {
                'ellipses_lotus_20': 'Lotus',
                'ellipses_lotus_limited_45': 'Lotus',
                'brain_walnut_120': 'Walnut',
                'ellipses_walnut_120': 'Walnut',
                'ellipsoids_walnut_3d': 'Walnut 3D',
                'ellipsoids_walnut_3d_60': 'Walnut 3D',
                }[data]
        data_title_part = 'Test on {}'.format(part)
    data_title_full = u'{} \u2013 {}'.format(data_title_dict[data], data_title_part)

    return data_title_full
