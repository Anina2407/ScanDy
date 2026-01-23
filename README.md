This repository collects my work of the LabRotation that based on the Framework ScanDy from Nicolas Rothe (Roth et al., 2023). 
I used the original framework but adapted at for images usage and solved logical and reusability issues. 

<p align="center">
  <img src="https://github.com/rederoth/ScanDy/blob/main/docs/scandy_repo_card.png">
</p>
<p align="center">
  <a href="https://github.com/psf/black">
  	<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://doi.org/10.1101/2023.03.14.532608">
    <img alt="paper" src="https://img.shields.io/badge/preprint-10.1101%2F2023.03.14.532608-blue"></a>    
</p>

## Introduction

`ScanDy` is a modular and mechanistic computational framework for simulating realistic **scan**paths in **dy**namic real-world scenes. It is specifically designed to quantitatively test hypotheses about eye-movement behavior in videos.
Specifically, it can be used to demonstrate the influence of object-representations on gaze behavior by comparing object-based and location-based models.

For a visual guide of how `ScanDy` works, have a look at the [interactive notebook](examples/interactive_guide.ipynb) (also on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/interactive_guide.ipynb)) and the <a href="#examples">example usecases</a>.

## Dataset

To prepare the dataset, we used the following resources:

* [VidCom](http://ilab.usc.edu/vagba/dataset/VidCom/) - Video and eye-tracking data
* [deep_em_classifier](https://github.com/MikhailStartsev/deep_em_classifier/) - Eye movement classification
* [detectron2](https://github.com/facebookresearch/detectron2/) - Frame-wise object segmentation
* [deep_sort](https://github.com/nwojke/deep_sort/) - Object tracking
* [dynamic-proto-object-saliency](https://github.com/csmslab/dynamic-proto-object-saliency/) - Low-level saliency maps
* [TASED-Net](https://github.com/MichiganCOG/TASED-Net/) - High-level saliency maps
* [PWC-Net](https://github.com/NVlabs/PWC-Net/) - Optical flow calculation

## Examples

The original examples from Nico can be still find in..

* [Example 1](examples/ex1_scanpath_sgl_video.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex1_scanpath_sgl_video.ipynb): Scanpath simulation and visualization for a single video
* [Example 2](examples/ex2_model_comparison.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex2_model_comparison.ipynb): Evolutionary optimization of model parameters
* [Example 3](examples/ex3_model_extension.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex3_model_extension.ipynb): Extending on existing models: Location-based model with object-based sensitivity


## More information

### The original Paper

> Roth, N., Rolfs, M., Hellwich, O., & Obermayer, K. (2023). Objects guide human gaze behavior in dynamic real-world scenes. *PLOS Computational Biology* 19(10): e1011512.

```bibtex
@article{roth2023objects,
 title = {Objects Guide Human Gaze Behavior in Dynamic Real-World Scenes},
 author = {Roth, Nicolas and Rolfs, Martin and Hellwich, Olaf and Obermayer, Klaus},
 journal = {PLOS Computational Biology},
 publisher = {Public Library of Science},
 year = {2023},
 month = {10},
 volume = {19},
 url = {https://doi.org/10.1371/journal.pcbi.1011512},
 pages = {1-39},
 number = {10},
```
