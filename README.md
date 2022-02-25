<div align="center">
  <h1>Curriculum-style Local-to-global Adaptation for Cross-domain Remote Sensing Image Segmentation<br></h1>
</div>

<!-- <div align="center">
  <h3><a href=></a>, <a href=></a>, <a href=></a>, <a href=></a></h3>
</div> -->

<div align="center">
  <h4> <a href=https://ieeexplore.ieee.org/document/9576523>[paper] IEEE Link</a></h4>
</div>

<div align="center">
  <h4> <a href=https://ieeexplore.ieee.org/document/9576523>[paper] ArXiv Link</a></h4>
</div>

<div align="center">
  <img src="./figure/2.png" width=800>
</div>


 <br><br/>
 
If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@article{zhang2021curriculum,
  title={Curriculum-Style Local-to-Global Adaptation for Cross-Domain Remote Sensing Image Segmentation},
  author={Zhang, Bo and Chen, Tao and Wang, Bin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--12},
  year={2021},
  publisher={IEEE}
}
```

## Code environment
This code requires Pytorch 1.7.1 and torchvision 0.8.2 or higher with cuda support. It has been tested on Ubuntu 18.04 

You can create a conda environment with the correct dependencies using the following command lines:
```
conda env create -f environment.yml
conda activate (need to add env name)
```

## Setting up data
Following [DualGAN](https://www.sciencedirect.com/science/article/pii/S0924271621000423), we crop the whole images in Potsdam IR-R-G dataset into the size of 512 × 512 with both horizontal and vertical strides of 512 pixels, and generate 4598 patches. For Vaihingen dataset, we crop the whole images into a size of 512 × 512 with both horizontal and vertical strides of 256 pixels and obtain 1696 patches

The following processed datasets are used in our paper: 
- PotsdamIRRG \[[Dataset Page](https://drive.google.com/file/d/1EuTBY25cq65KBYfCcCkcMqB0pMOQHGNw/view?usp=sharing)\]
- PotsdamRGB \[[Dataset Page](https://drive.google.com/file/d/1EuTBY25cq65KBYfCcCkcMqB0pMOQHGNw/view?usp=sharing)\]
- Vaihingen \[[Dataset Page](https://drive.google.com/file/d/1EuTBY25cq65KBYfCcCkcMqB0pMOQHGNw/view?usp=sharing)\]

After dowloading datasets, copy the data.zip to /CCDA_LGFA/ADVENT/ and extract it:
```
unzip data.zip
```

Note that the train/val split way has been saved in path: /CCDA_LGFA/ADVENT/advent/dataset/.

## Dowloading the source-domain pretrained model
- PotsdamIRRG pretrained weights \[[Dataset Page](https://drive.google.com/file/d/1JU3ZJmjwRJpUP5JuJhUNrGqKnovOGxSE/view?usp=sharing)\]
- Vaihingen pretrained weights \[[Dataset Page](https://drive.google.com/file/d/1fNm60XPvDyBJZ297TPEDmd1byaIQwmwd/view?usp=sharing)\]

We give an example of **PotsdamIRRG->Vaihingen adaptation direction**, and the other adaptation direction (e.g., Vaihingen->PotsdamIRRG) can be easily implemented by modifying the code

We provide a **two-stage curriculum adaptation strategy** for addressing the negative transfer issue within the target patches as follows:

## Before curriculum adaptation
- When the source-domain pretrained model is available, we can utilize this model to calculate the entropy scores of all target patches (or samples) as follows:

1. Change the inference batch_size to 1 in file: /CCDA_LGFA/ADVENT/advent/domain_adaptation/config.py
```
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cd /CCDA_LGFA/ADVENT/
pip install -e .
```

2. generate the easy/hard patches according to the source-domain pretrained model
```
cd /CCDA_LGFA/entropy_maps/ 
python entropy.py --lambda1 0.5
``` 

## Easy-to-adapt Training 
- Stage one: Adaptation the source-domain model from source domain to easy-to-adapt target patches

0. Change the inference batch_size to 4 in file: /CCDA_LGFA/ADVENT/advent/domain_adaptation/config.py
```
cfg.TRAIN.BATCH_SIZE_SOURCE = 4
cfg.TRAIN.BATCH_SIZE_TARGET = 4
```
1. Training in feature-level adaptation of the Stage one
```
cd /CCDA_LGFA/ADVENT/advent/scripts/
```
```
python train_easy.py --status 0 --cfg /root/code/CCDA_LGFA/ADVENT/advent/scripts/configs/advent_easy.yml # path for advent_easy.yml
```

2. Evaluation in feature-level adaptation of the Stage one
```

python test.py --status 0 --cfg /root/code/CCDA_LGFA/ADVENT/advent/scripts/configs/advent_easy.yml
```
3. Change EXP_NAME and RESTORE_FROM (using the best model in the above stage) in advent_easy.yml, and training in entropy-level adaptation of the Stage one
```
python train_easy.py --status 1 --cfg /root/code/CCDA_LGFA/ADVENT/advent/scripts/configs/advent_easy.yml # path for advent_easy.yml
```

4. Evaluation in entropy-level adaptation of the Stage one
```
python test.py --status 1 --cfg /root/code/CCDA_LGFA/ADVENT/advent/scripts/configs/advent_easy.yml
```


## Before Hard-to-adapt Training
Re-evaluate the adaptation difficulty of the all target patches as follow:

1. Change the inference batch_size to 1 in file: /CCDA_LGFA/ADVENT/advent/domain_adaptation/config.py
```
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
```

2. generate the easy/hard patches according to the SL-Adapted-Baseline model
```
cd /CCDA_LGFA/entropy_maps/ 
Replace the "restore_from" in entropy.py using the best model in the above stage
python entropy.py --lambda1 0.5
``` 

3. cd /CCDA_LGFA/ADVENT/advent/dataset/PotsdamIRRG/ and create the train_and_pseudo_r_0.5.txt file of pseudo-labeled target samples and labeled source samples as follows:
```
vim train_and_pseudo_r_0.5.txt, where this file contains all lines of both train.txt (PotsdamIRRG) and easy_split.txt (Vaihingen)
``` 

4. Change the "cfg.TRAIN.SET_SOURCE" to 'train_and_pseudo_r_0.5' in /CCDA_LGFA/ADVENT/advent/domain_adaptation/config.py
5. Copy all easy-to-adapt images to /CCDA_LGFA/ADVENT/data/PotsdamIRRG/images, according to the easy_split.txt
6. Copy pseudo-labels of easy-to-adapt images (which saved in /CCDA_LGFA/entropy_maps/color_masks/) to /CCDA_LGFA/ADVENT/data/PotsdamIRRG/labels/ , according to the easy_split.txt


## Hard-to-adapt Training
- Stage two: Adaptation the initially-aligned model from source and pseudo-labeled domains to hard-to-adapt target patches
0. Change the inference batch_size to 4 in file: /CCDA_LGFA/ADVENT/advent/domain_adaptation/config.py
```
cfg.TRAIN.BATCH_SIZE_SOURCE = 4
cfg.TRAIN.BATCH_SIZE_TARGET = 4
```
1. Training in feature-level adaptation in Stage two
```
cd /CCDA_LGFA/to_hard/
```
```
python train_patch_wise_hard.py --status 0 --cfg /root/code/CCDA_LGFA/to_hard/to_hard.yml # path for to_hard.yml
```

2. Evaluation in feature-level adaptation in Stage two
```
python test_patch_wise.py --status 0 --cfg /root/code/CCDA_LGFA/to_hard/to_hard.yml # path for to_hard.yml
```
3. Change EXP_NAME and RESTORE_FROM (using the best model in the above stage) in to_hard.yml, and training in entropy-level adaptation in Stage two:
```
python train_patch_wise_hard.py --status 1 --cfg /root/code/CCDA_LGFA/to_hard/to_hard.yml # path for to_hard.yml
```

4. Evaluation in entropy-level adaptation in Stage two
```
python test_patch_wise.py --status 1 --cfg /root/code/CCDA_LGFA/to_hard/to_hard.yml # path for to_hard.yml
```

## Selected cross-domain semantic segmentation results
Here we report some performance comparisons from our paper on PotsdamIRRG, PotsdamIRRG, and Vaihingen

<div align="center">
  <img src="./figure/table_1.png" width=800>
</div>

<div align="center">
  <img src="./figure/table_2.png" width=800>
</div>

<div align="center">
  <img src="./figure/table_3.png" width=800>
</div>

## Contact
We have tried our best to verify the correctness of our released data, code and trained model weights. 
However, there are a large number of experiment settings, all of which have been extracted and reorganized from our original codebase. 
There may be some undetected bugs or errors in the current release. 
If you encounter any issues or have questions about using this code, please feel free to contact us via bo.zhangzx@gmail.com

