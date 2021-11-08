import os
import json
from evaluation.display_utils import get_title_from_run_spec

EVAL_RESULTS_PATH = os.path.dirname(__file__)

eval_results_filename = 'comparison_on_ellipses_lotus_20_all'
# eval_results_filename = 'comparison_on_ellipses_lotus_limited_45_all'
# eval_results_filename = 'comparison_on_brain_walnut_120_all'
# eval_results_filename = 'comparison_on_ellipses_walnut_120_all'

tex = ''

# rise_time_fmt = '{:d}'
# psnr_steady_fmt = '{:.2f}'
# psnr_0_fmt = '{:.2f}'
rise_time_fmt = '\\num{{{:d}}}'
psnr_steady_fmt = '\\num{{{:.2f}}}'
psnr_0_fmt = '\\num{{{:.2f}}}'

with open(os.path.join(
        EVAL_RESULTS_PATH, '{}.{}'.format(eval_results_filename, 'json')),
        'r') as f:
    eval_results_list = json.load(f)

    for eval_results in eval_results_list:
        title = get_title_from_run_spec(eval_results['run_spec'])
        tex += '  '
        tex += title
        tex += ' & '
        tex += (rise_time_fmt.format(eval_results['rise_time_to_baseline'])
                if eval_results['rise_time_to_baseline'] is not None else '-')
        tex += ' & '
        tex += psnr_steady_fmt.format(eval_results['PSNR_steady'])
        tex += ' & '
        tex += psnr_0_fmt.format(eval_results['PSNR_0'])
        tex += '\\\\\n'

print('Eval results in file \'{}\' as as TeX tabular rows:'.format(
        eval_results_filename))
print(tex)
