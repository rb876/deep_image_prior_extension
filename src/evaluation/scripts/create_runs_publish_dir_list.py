import os
import yaml

# after running this script, copy the results using (e.g.):
# rsync -r --files-from=runs_publish_dir_list_ellipses_lotus_20.txt /path/to/src/ /path/to/dest/

# data_to_publish = ''
data_to_publish = 'ellipses_lotus_20'
# data_to_publish = 'ellipses_lotus_limited_45'
# data_to_publish = 'brain_walnut_120'
# data_to_publish = 'ellipses_walnut_120'

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'runs_publish.yaml'), 'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

run_paths = []
for kind, runs_kind in runs.items():
    for data, runs_data in runs_kind.items():
        if data == data_to_publish:
            for experiment, runs_experiment in runs_data.items():
                for run in runs_experiment:
                    run_path = run.get('run_path', run.get('single_run_path'))
                    run_paths.append(run_path)

filename = 'runs_publish_dir_list{}.txt'.format(
        '' if not data_to_publish else '_{}'.format(data_to_publish))

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       filename), 'w') as f:
    for run_path in run_paths:
        f.write(run_path + '\n')
