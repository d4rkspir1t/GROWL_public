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
feats = 20
epochs = 100
balanced = '-b'
string_end = '20220118'
ablations = ['no', 'ori', 'edg']
ablation = ablations[0]
test_on_options = ['ps', 'cpp', 'all_salsa', 'rica', 'rica_yolo']
test_on = test_on_options[0]
iter = 0
tracker_file_path = 'growl_param_analysis/no_ablation_ps_test_20_feats_100_epochs_1_balanced_model_f1output_20220118.csv'

while True:
    print("\nStarting " + filename)
    func_call = 'python %s -f %d -e %d -s %s -t %s -a %s %s' % (filename, feats, epochs, string_end, test_on, ablation, balanced)
    p = Popen(func_call, shell=True)
    p.wait()
    time.sleep(60)
    print('DONE ', iter, ' STEPS')

    df = pd.read_csv(tracker_file_path)
    print('DF SHAPE', df.shape[0])
    if df.shape[0] == 30:
        break