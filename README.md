# Seismic-denoising-and-interpolation-based-on-Spatial-Multi-scale-CSWinT
This repository gives the codes for "Simultaneous Denoising and Interpolation of Seismic Data based on Spatial Multi-scale Cross-shaped Window Transformer".

Step 1: Generate synthetic seismic. 
Generate synthetic seismic data and store the training data set in "data/train" and the test set in "data/test".

Step 2: Training patches dataset. 
Please run the train.py file directly.

Step 3: Test the model. 
Choose an optimal model to use on the test data. Please run the test.py file directly.


model: models of network design.

data/test: synthetic data for the test.

data/train: synthetic data for training is obtained from https://wiki.seg.org/wiki/Open_data.

output/net_smcswt.pth: trained model.

modelds/network: network for training and testing.

modelds/util: tools for training and testing.

test.py: test code.

train.py: training code.

Due to the sensitive nature of the commercial datasets, the raw data would remain confidential and would not be shared.
