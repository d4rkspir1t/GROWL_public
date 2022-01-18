# ---------------------------------------------------------------------------
# Created By  : Viktor Schmuck - https://github.com/d4rkspir1t
# Created Date: 07/01/2021
# Last revised: 17/01/2022
# ---------------------------------------------------------------------------
__author__ = 'Viktor Schmuck'
__credits__ = ['Viktor Schmuck', 'Oya Celiktutan']
__license__ = 'MIT'
__version__ = '1.0.1'
__maintainer__ = 'Viktor Schmuck'
__email__ = 'viktor.schmuck@kcl.ac.uk'
__status__ = 'Development in Progress'

from subprocess import Popen
import sys
import time

import pandas as pd

# TODO: update call with all params
filename = 'growl_test_code.py'
ablations = ['no', 'ori', 'edg']
ablation = ablations[0]
test_on_options = ['ps', 'cpp', 'all_salsa', 'rica', 'rica_yolo']
test_on = test_on_options[0]
iter = 0
tracker_file_path = 'growl_param_analysis/NO_ablation_SALSAPS_test_20_feats_100_epochs_model_f1output_20220113.csv'

while True:
    print("\nStarting " + filename)
    func_call = 'python %s --test %s --ablation %s' % (filename, test_on, ablation)
    p = Popen(func_call, shell=True)
    p.wait()
    time.sleep(60)
    print('DONE ', iter, ' STEPS')

    df = pd.read_csv(tracker_file_path)
    print('DF SHAPE', df.shape[0])
    if df.shape[0] == 30:
        break