# IRFA: Inverse receptive field attention for naturalistic image reconstruction from the brain
_For the supplementary material and source code of the IRFA project
Abstract: Visual perception in the brain largely depends on the organization of neuronal receptive fields. Although extensive research has delineated the coding principles of receptive fields, most studies have been constrained by their foundational assumptions. Moreover, while machine learning has successfully been used to reconstruct images from brain data, this approach faces significant challenges, including inherent feature biases in the model and the complexities of brain structure and function. In this study, we introduce an inverse receptive field attention (IRFA) model, designed to reconstruct naturalistic images from neurophysiological data in an end-to-end fashion. This approach aims to elucidate the tuning properties and representational transformations within the visual cortex. The IRFA model incorporates an attention mechanism that determines the inverse receptive field for each pixel, weighting neuronal responses across the visual field and feature spaces. This method allows for an examination of the dynamics of neuronal representations across stimuli in both spatial and feature dimensions. Our results show highly accurate reconstructions of naturalistic data, independent of pre-trained models. Notably, IRF models trained on macaque V1, V4, and IT regions yield remarkably consistent spatial receptive fields across different stimuli, while the features to which neuronal representations are selective exhibit significant variation. Additionally, we propose a data-driven method to explore representational clustering within various visual areas, further providing testable hypotheses.


_


## Intro
In this repository, you will find the following:
Source code:
- training folder: the `IRFA.py` contains the IRFA model and the loss function. The training loop can be found in `train.py` and the `discriminator.py` contains the architecture for the discriminator.
- analysis folder: `quantification.ipynb` shows how the reconstructions were quantified and `spatial_IRF.ipny` and `feature_IRF.ipny` how the spatial and feature inverse receptive fields (IRFs) were visualized respectively.
  
