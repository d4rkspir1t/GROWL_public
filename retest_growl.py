from subprocess import Popen
import sys
import time
import pandas as pd

filename = 'growl_test_code.py'
iter = 0
tracker_file_path = 'growl_param_analysis/NO_ablation_SALSAPS_test_20_feats_100_epochs_model_f1output_20220113.csv'

while True:
    print("\nStarting " + filename)
    p = Popen("python " + filename, shell=True)
    p.wait()
    # iter += 1
    time.sleep(60)
    print('DONE ', iter, ' STEPS')

    df = pd.read_csv(tracker_file_path)
    print('DF SHAPE', df.shape[0])
    if df.shape[0] == 30:
        break
    # if iter == 10:
    #     break