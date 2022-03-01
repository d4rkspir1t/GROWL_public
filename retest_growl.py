# ---------------------------------------------------------------------------
# Created By  : Viktor Schmuck - https://github.com/d4rkspir1t
# Created Date: 07/01/2021
# Last revised: 21/01/2022
# ---------------------------------------------------------------------------
__author__ = 'Viktor Schmuck'
__credits__ = ['Viktor Schmuck', 'Oya Celiktutan']
__license__ = 'MIT'
__version__ = '1.0.1'
__maintainer__ = 'Viktor Schmuck'
__email__ = 'viktor.schmuck@kcl.ac.uk'
__status__ = 'Development in Progress'

import logging
from subprocess import Popen, STDOUT, PIPE
import sys
import time

import pandas as pd

logger = logging.getLogger()
logger.setLevel('INFO')


def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logging.info('subprocess: %r', line)


filename = 'CCP_ensemble_code.py'
feats = 20
epochs = 100
balanced = 3
string_end = '20220228_balance_3_try_3_CCP_10folds'
ablations = ['no', 'ori', 'edg']
ablation = ablations[0]
test_on_options = ['ps', 'cpp', 'all_salsa', 'rica', 'rica_yolo']
test_on = test_on_options[1]
test_iters = 30
tracker_file_path = 'growl_param_analysis/%s_ablation_%s_test_%d_feats_%d_epochs_%d_balanced_model_f1output_%s.csv' % \
                        (ablation, test_on, feats, epochs, balanced, string_end)
while True:
    print("\nStarting " + filename)
    func_call = 'python %s -f %d -e %d -s %s -t %s -a %s -b %d' % (filename, feats, epochs, string_end, test_on, ablation, balanced)
    p = Popen(func_call, shell=True, stdout=PIPE, stderr=STDOUT)
    with p.stdout:
        log_subprocess_output(p.stdout)
    p.wait()
    time.sleep(60)

    df = pd.read_csv(tracker_file_path)
    print('DONE ', df.shape[0], ' STEPS')
    # print('DF SHAPE', df.shape[0])
    if df.shape[0] == test_iters:
        break