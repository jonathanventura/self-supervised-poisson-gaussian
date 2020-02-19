# Self-Supervised Poisson-Gaussian Denoising

Code for our paper "Self-Supervised Poisson-Gaussian Denoising".

# Dependencies

* Python 3.6
* Tensorflow
* scikit-image
* Keras
* tqdm

# Dataset download

    bash download_fmd.sh

# Training

To train various models on the Confocal Mice dataset:

    python train_fmd.py --path ./dataset --dataset Confocal_MICE --mode uncalib ;
    python train_fmd.py --path ./dataset --dataset Confocal_MICE --mode poissongaussian --reg 1 ;

To test the trained models:

    python test_fmd.py --path ./dataset --dataset Confocal_MICE --mode uncalib ; 
    python test_fmd.py --path ./dataset --dataset Confocal_MICE --mode poissongaussian --reg 1 ; 

