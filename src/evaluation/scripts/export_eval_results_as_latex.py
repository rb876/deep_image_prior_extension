import os
import json
from evaluation.display_utils import get_title_from_run_spec

EVAL_RESULTS_PATH = os.path.dirname(__file__)

# eval_results_filename = 'comparison_on_ellipses_lotus_20_all'
# eval_results_filename = 'comparison_on_ellipses_lotus_limited_45_all'
# eval_results_filename = 'comparison_on_brain_walnut_120_all'
# eval_results_filename = 'comparison_on_ellipses_walnut_120_all'
# eval_results_filename = 'comparison_on_ellipsoids_walnut_3d_all'
eval_results_filename = 'comparison_on_ellipsoids_walnut_3d_60_all'

tex = ''

rise_time_fmt = '\\num{{{:04d}}}'
psnr_steady_fmt = '\\num{{{:.2f}}}'
psnr_best_fmt = '$\\num{{{:.2f}}}|_{{i=\\num{{{:04d}}}}}$'
psnr_0_fmt = '\\num{{{:.2f}}}'

def format_rise_time(x):
    if x is None:
        return '-'
    out = '\\num{{{:d}}}'.format(x)
    if 'walnut' in eval_results_filename and x < 10000:
        out = '\\hphantom{0\\,}' + out
    if x < 1000:
        out = '\\hphantom{0}' + out
    return out

def format_psnr_steady(x):
    return '\\num{{{:.2f}}}'.format(x)

def format_psnr_best(x, iteration):
    out_iteration = '\\num{{{:04d}}}'.format(iteration)
    if 'walnut' in eval_results_filename and iteration < 10000:
        out_iteration = '\\hphantom{0\\,}' + out_iteration
    if iteration < 1000:
        out_iteration = '\\hphantom{0}' + out_iteration
    return '$\\num{{{:.2f}}}|_{{i={}}}$'.format(x, out_iteration)

def format_psnr_0(x):
    out = '\\num{{{:.2f}}}'.format(x)
    if x < 10.:
        out = '\\hphantom{0}' + out
    return out

with open(os.path.join(
        EVAL_RESULTS_PATH, '{}.{}'.format(eval_results_filename, 'json')),
        'r') as f:
    eval_results_list = json.load(f)

    for eval_results in eval_results_list:
        title = get_title_from_run_spec(eval_results['run_spec'])
        tex += '  '
        tex += title
        tex += ' & '
        tex += format_rise_time(eval_results['rise_time_to_baseline'])
        tex += ' & '
        tex += format_psnr_best(eval_results['PSNR_best'], eval_results['PSNR_best_iter'])
        tex += ' & '
        tex += psnr_steady_fmt.format(eval_results['PSNR_steady'])
        tex += ' & '
        tex += format_psnr_0(eval_results['PSNR_0'])
        tex += '\\\\\n'

print('Eval results in file \'{}\' as TeX tabular rows:'.format(
        eval_results_filename))
print(tex)
