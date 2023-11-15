# Generalization estimation in deep neural networks

This work is for estimating the generalization gap using only training set. 

The example.py file provides an example of the whole implementation, mainly consisting of two parts. The first part is model training (VGG16 is provided here). One could freely specify various hyper-parameters in the .json file in the model directory for obtaining models with different generalization gap. The given json file provides parameters of six models with different fractions of corrupted labels from 0 to 1 with 0.2 interval. The second part is the estimation of generalization gap. One should pass the trained networks to the estimation module.
