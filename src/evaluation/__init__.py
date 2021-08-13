"""
As a convention of this module, `run_spec` dictionaries are used that both
select (runs from ) a (multi-)run out of those listed in `runs.yaml` and
specify how it should be displayed.
Fields of a `run_spec` dictionary include:

    experiment : str
        Experiment name
    sub_runs : iterable of int, optional
        Indices of sub-runs to select from the multirun (`None` selects all)
    name : str, optional
        Name of the run given in ``runs.yaml`` or `None` to select the first
    name_title : str, optional
        Display title for name (defaults to ``run_spec['name']``)
    title_prefix : str, optional
        Prefix for title
    title_suffix : str, optional
        Suffix for title
    experiment_title : str, optional
        Display title for experiment (defaults to
        ``experiment_title_dict[run_spec['experiment']]``)
"""

from .utils import *
