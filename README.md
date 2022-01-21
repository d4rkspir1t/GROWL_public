# GROWL_public
Repository holding the implementation for GROWL - Group Detection With Link Prediction

## References
When using this code and/or the RICA dataset, please cite the following works:
#### RICA: Robocentric Indoor Crowd Analysis Dataset
```
@article{schmuck2020rica,
  title={RICA: Robocentric Indoor Crowd Analysis Dataset},
  author={Schmuck, Viktor and Celiktutan, Oya},
  journal={IMU},
  volume={127},
  number={74,234},
  pages={31--172},
  year={2020}
}
```

####  GROWL: Group Detection With Link Prediction
```
@inproceedings{schmuck2021growl,
  title={GROWL: Group Detection With Link Prediction},
  author={Schmuck, Viktor and Celiktutan, Oya},
  booktitle={2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

## Installation
* Download or clone the repository
* Set up an environment with python 3.9
* Install the following packages (indicating latest tested compatible versions):
  * pandas==1.3.5
  * matplotlib==3.5.1
  * scipy==1.7.3
  * numpy==1.22.1
  * dgl==0.6.1
  * torch==1.10.1
  * networkx==2.6.3
  * scikit-learn==1.0.2

### Quick installation commands if you are using conda
```
conda create --name growl_pub python=3.9
pip install pandas matplotlib scipy numpy dgl torch networkx scikit-learn
```

## Usage
For multiple tests, run `retest_growl.py`, changing the parameters in lines 20-33 as you see fit. 
These are described below:
* `feats`: number of features to create during embedding
* `epochs`: number of epoch to train for
* `balanced`: whether only training runs with balanced negative and positive edge samples 
should be executed. Set to `balanced=''` if not.
* `string_end`: how the output csv filename should end, usually a date
* `ablation`: if an ablation test should be executed. Choose a corresponding index from the `ablations` list.
* `test_on`: choose which dataset the tests should be executed on. Choose a corresponding index from 
the `test_on_options` list.
* `test_iters`: how many iterations of the test should be executed.

---

It's also possible to execute individual tests by calling (e.g.):
```
python growl_test_code.py -f 20 -e 100 -s 20220121 -t all_salsa -a no -b
```

Command line parameter usage:
```
usage: growl_test_code.py [-h] [-f FEATS] [-e EPOCHS] [-s STRING_END] [-t {ps,cpp,all_salsa,rica,rica_yolo}] [-a {no,ori,edg}] [-b] [-p] [--yolo_acc] [--yolo_to_gt]

To perform tests with GROWL, decide which data to test on and whether this is an ablation test.

optional arguments:
  -h, --help            show this help message and exit
  -f FEATS, --feats FEATS
                        Sets the number of features to create an embedding for.
  -e EPOCHS, --epochs EPOCHS
                        Sets the numebr of epochs to train GROWL for.
  -s STRING_END, --string_end STRING_END
                        The postfix of the results filename.
  -t {ps,cpp,all_salsa,rica,rica_yolo}, --test {ps,cpp,all_salsa,rica,rica_yolo}
                        Which dataset to test on.
  -a {no,ori,edg}, --ablation {no,ori,edg}
                        Is this a test involving ablation?
  -b, --balance_samples
                        Enable sample balancing to discard runs if there is too big of an imbalance between positive and negative edge counts.
  -p, --plot            Enable plotting the graphs produced during the tests.
  --yolo_acc            Print YOLOv4 detection accuracy compared to RICA's ground truth. Only has an effect if test=rica_yolo.
  --yolo_to_gt          When testing on YOLOv4 detections, calculate F1-scores of detected groups compared to the GT detections.

```