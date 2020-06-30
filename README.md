# HQS-Net: A Neural Network-based Reconstruction in Compressed Sensing MRI Without Fully-sampled Training Data
<img src="figures/architecture.png" width='600'>

## Abstract
Compressed Sensing MRI (CS-MRI) has shown promise in reconstructing under-sampled MR images, offering the potential to reduce scan times. Classical techniques minimize a regularized least-squares cost function using an expensive iterative optimization procedure. Recently, deep learning models have been developed that model the iterative nature of classical techniques by unrolling iterations in a neural network. While exhibiting superior performance, these methods require large quantities of ground-truth images and have shown to be non-robust to unseen data. In this paper, we explore a novel strategy to train an unrolled reconstruction network in an unsupervised fashion by adopting a loss function widely-used in classical optimization schemes. We demonstrate that this strategy achieves lower loss and is computationally cheap compared to classical optimization solvers while also exhibiting superior robustness compared to supervised models.

## Requirements
The code was tested on:
- python 3.6
- pytorch 1.1
- torchvision 0.3.0
- scikit-image 0.15.0
- scikit-learn 0.19.1
- matplotlib 3.0.2
- numpy 1.15.4
- tqdm 4.38.0

## Instruction

### Preprocessing 
T1-weighted 3D brain MRI scans was preprocessed using FreeSurfer [1]. This includes skull stripping, bias-field correction, intensity normalization, affine registration to Talairach space, and resampling to 1 mm<sup>3</sup> isotropic resolution. The original images had a shape of 256x256x256, which was futher cut using ((48, 48), (31, 33), (3, 29)) along the sides to elimate empty space. 

Unfortunately, Buckner40 and its manual segmentation are not public dataset. However, we provide some example OASIS [2] volumes in `./data/vols/` and automatic segmentations in `./data/labels/`. The labels legend can be found in the [FreeSurfer webpage](https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT). We also provide the probabilistic atlas that we used in our experiment in the data folder. Check `functions/dataset.py` for further details on how we convert these numpy files into torch tensor. 

### Training
`python train_network.py --undersample_rate 4p2 --dataset t1`

### Evaluation
Run `python test.py 0`. It will return a dictionary which can be loaded using `torch.load(result.pth.tar')['stat']`. It has 13 columns. The 12 first column contain the dice loss (1-dice) of 12 region of interest (rois). Namely: pallidum, amygdala, caudate, cerebral cortex, hippocampus, thalamus, putamen, cerebral wm, cerebellum cortex, lateral ventricle, cerebellum wm and brain-stem. The last column is the average of the rois.

## Contact
Feel free to open an issue in github for any problems or questions.

## References
[1] Bruce Fischl. Freesurfer. Neuroimage, 62(2):774–781, 2012

[2] Marcus et al. Open access series of imaging studies (OASIS): cross-sectional MRI data
in young, middle aged, nondemented, and demented older adults. Journal of Cognitive
Neuroscience, 2007.
