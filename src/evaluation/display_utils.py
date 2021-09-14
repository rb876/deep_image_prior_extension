"""
Utilities for displaying results (e.g. consistent titles).
"""

experiment_title_dict = {
    'pretrain': 'Pretrained DIP (switch to noise)',
    'pretrain_only_fbp': 'Pretrained DIP (FBP)',
    'pretrain_noise': 'Pretrained DIP (FBP & noise)',
    'no_pretrain': 'DIP (noise)',
    'no_pretrain_fbp': 'DIP (FBP)',
    'no_pretrain_2inputs': 'DIP (FBP & noise)',
}

data_title_dict = {
    'ellipses_lotus_20': 'Ellipses/Lotus Sparse 20',
    'ellipses_lotus_40': 'Ellipses/Lotus Sparse 40',
    'ellipses_lotus_limited_30': 'Ellipses/Lotus Limited 30',
}

experiment_color_dict = {
    'pretrain_only_fbp': '#EC2215',
    'pretrain': '#B15CD1',
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
    data_title_full_parts = [data_title_dict[data]]
    data_title_full_parts.append('Validation (Shepp-Logan)' if validation_run
                                 else 'Test (Lotus)')

    data_title_full = ', '.join(data_title_full_parts)

    return data_title_full
