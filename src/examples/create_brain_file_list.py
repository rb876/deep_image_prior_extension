"""
Create a list of dcm files for :class:`dataset.brain.ACRINFMISOBrainDataset`.

Requirement: ACRIN-FMISO-Brain dataset stored at `DATA_PATH`
https://doi.org/10.7937/K9/TCIA.2018.vohlekok

Note: The NBIA data retriever seems to produce different directory names when
using the linux command line interface (CLI)
https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command+Line+Interface
compared to using the GUI app.  This script works for both, but the directory
and file lists are specific to the respective version.  In this repo, file for
both the CLI and the GUI version are included:

    ``data/brain/brain_axial_min_512px_non_mosaic_mr_dirs.json``
    ``data/brain/brain_axial_min_512px_non_mosaic_mr_dirs_gui.json``
    ``data/brain/acrin_fmiso_brain_file_list.json``
    ``data/brain/acrin_fmiso_brain_file_list_gui.json``

The dataset was downloaded with NBIA data retriever 4.1 using
``src/dataset/ACRIN-FMISO-Brain July 2 2019 NBIA manifest.tcia`` in Sep 2021.
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from pydicom.filereader import dcmread
import json
from tqdm import tqdm

DATA_PATH = '/localdata/ACRIN-FMISO-Brain'

def is_valid_dataset(dataset):
    if dataset.Modality != 'MR':
        return False
    if max(abs(dataset.ImageOrientationPatient[2]),
            abs(dataset.ImageOrientationPatient[5])) >= np.cos(np.pi / 4):
        return False  # not axial
    if max(int(dataset.Rows), int(dataset.Columns)) < 512:
        return False
    if 'MOSAIC' in (str(t) for t in dataset.get('ImageType', [])):
        return False
    return True

def get_dirs():
    """Return the list of directories to include in the dataset.

    The dicom files in each included directory are all valid, as determined by
    :func:`is_valid_dataset`. Directories that also contain other dicom files,
    e.g. with a different orientation, are omitted.
    """
    ct_dirs = []
    for root, _, files in tqdm(os.walk(DATA_PATH),
                               total=sum(1 for _ in os.walk(DATA_PATH))):
        dcm_files = sorted([f for f in files if f.endswith('.dcm')])
        if len(dcm_files) >= 1:
            if all(is_valid_dataset(dcmread(os.path.join(root, f)))
                   for f in dcm_files):
                ct_dirs.append(os.path.relpath(root, DATA_PATH))
    ct_dirs.sort()

    return ct_dirs

def get_num_slices_per_patient(mr_dirs):
    num_slices_per_patient = {}

    for mr_dir in mr_dirs:
        dcm_files = [f for f in os.listdir(os.path.join(DATA_PATH, mr_dir))
                     if f.endswith('.dcm')]
        patient = mr_dir.split('/')[0]
        num_slices_per_patient.setdefault(patient, 0)
        num_slices_per_patient[patient] += len(dcm_files)

    num_slices_per_patient = {
            k: v for k, v in sorted(num_slices_per_patient.items())}

    return num_slices_per_patient

def find_split(mr_dirs,
               num_validation_patients=4, num_test_patients=4,
               validation_fraction_range=(0.09, 0.13),
               test_fraction_range=(0.09, 0.13),
               verbose=True, rng=None):

    patients = list(np.unique([mr_dir.split('/')[0] for mr_dir in mr_dirs]))

    split = {}

    num_slices_per_patient = get_num_slices_per_patient(mr_dirs)
    num_slices_total = sum(num_slices_per_patient.values())

    if validation_fraction_range[0] is None:
        validation_fraction_range[0] = 0.
    if test_fraction_range[0] is None:
        test_fraction_range[0] = 0.
    if validation_fraction_range[1] is None:
        validation_fraction_range[1] = 1.
    if test_fraction_range[1] is None:
        test_fraction_range[1] = 1.

    if rng is None:
        rng = np.random.default_rng()

    reject = True
    while reject:
        patients_copy = patients.copy()
        rng.shuffle(patients_copy)
        split['test'] = patients_copy[-num_test_patients:]
        del patients_copy[-num_test_patients:]
        split['validation'] = patients_copy[-num_validation_patients:]
        del patients_copy[-num_validation_patients:]
        split['train'] = patients_copy

        test_fraction = (
                sum((num_slices_per_patient[p] for p in split['test'])) /
                num_slices_total)
        validation_fraction = (
                sum((num_slices_per_patient[p] for p in split['validation'])) /
                num_slices_total)
        reject = (test_fraction < test_fraction_range[0] or
                  test_fraction > test_fraction_range[1] or
                  validation_fraction < validation_fraction_range[0] or
                  validation_fraction > validation_fraction_range[1])

        if verbose:
            print('rejecting split with '
                  'test_fraction={:f} (accept range is [{:f}, {:f}]), '
                  'validation_fraction={:f} (accept range is [{:f}, {:f}])'
                  .format(test_fraction, *test_fraction_range,
                          validation_fraction, *validation_fraction_range))

    if verbose:
        print('accepting split with '
                'test_fraction={:f} (accept range is [{:f}, {:f}]), '
                'validation_fraction={:f} (accept range is [{:f}, {:f}])'
                .format(test_fraction, *test_fraction_range,
                        validation_fraction, *validation_fraction_range))

    return split

if __name__ == '__main__':
    mr_dirs = get_dirs()
    with open('brain_axial_min_512px_non_mosaic_mr_dirs.json', 'w') as f:
        json.dump(mr_dirs, f, indent=1)
    # with open('brain_axial_min_512px_non_mosaic_mr_dirs.json', 'r') as f:
    #     mr_dirs = json.load(f)
    print('number of MR directories:', len(mr_dirs))

    patients = list(np.unique([s.split('/')[0] for s in mr_dirs]))

    print('number of patients:', len(patients))
    print(patients)

    num_slices_per_patient = get_num_slices_per_patient(mr_dirs)

    print('number of slices per patient:', num_slices_per_patient)
    print('total number of slices:',
          sum(n for n in num_slices_per_patient.values()))

    split = find_split(mr_dirs, rng=np.random.default_rng(seed=8))

    fold_per_patient = {}
    for fold, patients in split.items():
        for p in patients:
            fold_per_patient[p] = fold

    file_list = {'train': [], 'validation': [], 'test': []}

    for mr_dir in tqdm(mr_dirs):
        patient = mr_dir.split('/')[0]
        fold = fold_per_patient[patient]
        file_list[fold] += [
                os.path.join(mr_dir, f)
                for f in os.listdir(os.path.join(DATA_PATH, mr_dir))
                if f.endswith('.dcm')]

    for fold in file_list.keys():
        file_list[fold].sort()

    with open('acrin_fmiso_brain_file_list.json', 'w') as f:
        json.dump(file_list, f, indent=1)


    # check that MR dirs lists are identical for CLI and GUI version up to
    # renaming

    # def convert_path_cli_to_gui(path):
    #     replacements = [
    #         (' WO-W ', ' WOW '),
    #         (' W-WO ', ' WWO '),
    #         (' w-wo ', ' wwo '),
    #         (' w- ', ' w '),
    #         (' w-o ', ' wo '),
    #         ('_', ''),
    #         ('*', ''),
    #         ('(', ''),
    #         (')', ''),
    #         ('[', ''),
    #         (']', ''),
    #         (':', ''),
    #         ('+', ''),
    #         ('^', ''),
    #         ('&', ''),
    #         ('-MRIBRNWWO-PER-3D-', '-MRIBRNWWOPER3D-'),
    #         ('-MRIBRNWWO-PER-', '-MRIBRNWWOPER-'),
    #         ('-MRIBRNWWO-PER-WR-', '-MRIBRNWWOPERWR-'),
    #         ('-MRIBRNWWO-3D-PER-WR-', '-MRIBRNWWO3DPERWR-'),
    #         ('-MRIBRNWWO-3D-WR-PER-', '-MRIBRNWWO3DWRPER-'),
    #         ('-MRIBRNWWO-3D-WR-', '-MRIBRNWWO3DWR-'),
    #         (' F-SAT+GAD-', ' FSATGAD-'),
    #         (' F-SATGAD-', ' FSATGAD-'),
    #         (' GAD-19ML ', ' GAD19ML '),
    #         ('-NEURO- ', '-NEURO '),
    #         ('-MRI  BRAIN ', '-MRI BRAIN '),
    #         (' W-DIFF-', ' WDIFF-'),
    #     ]
    #     for old, new in replacements:
    #         path = path.replace(old, new)
    #     return path

    # with open('data/brain/brain_axial_min_512px_non_mosaic_mr_dirs.json', 'r') as f:
    #     mr_dirs = json.load(f)
    # with open('data/brain/brain_axial_min_512px_non_mosaic_mr_dirs_gui.json', 'r') as f:
    #     mr_dirs_gui = json.load(f)

    # assert len(mr_dirs) == len(mr_dirs_gui)

    # for mr_dir, mr_dir_gui in zip(mr_dirs, mr_dirs_gui):
    #     assert (convert_path_cli_to_gui(mr_dir) == mr_dir_gui), (
    #             "{} !=\n{}".format(convert_path_cli_to_gui(mr_dir), mr_dir_gui))


    # show numbers of slices for each (maximum) image size

    # num_slices_per_max_size = {}

    # for mr_dir in tqdm(mr_dirs):
    #     dcm_files = [f for f in os.listdir(os.path.join(DATA_PATH, mr_dir))
    #                  if f.endswith('.dcm')]
    #     dcm_file = dcm_files[len(dcm_files) // 2]
    #     dataset = dcmread(os.path.join(DATA_PATH, mr_dir, dcm_file))
    #     max_size = max(int(dataset.Rows), int(dataset.Columns))
    #     num_slices_per_max_size.setdefault(max_size, 0)
    #     num_slices_per_max_size[max_size] += len(dcm_files)

    # num_slices_per_max_size = {
    #         k: v for k, v in sorted(num_slices_per_max_size.items())}

    # print(num_slices_per_max_size)
