This repository collects my work of the LabRotation that based on the Framework ScanDy from Nicolas Rothe (Roth et al., 2023). <br>
I used the original framework but adapted at for images usage and solved logical and reusability issues. 


## The Framework - ScanDy
![differentmodules_on_pictures_slow](https://github.com/user-attachments/assets/438f8e6c-f885-48fb-9056-495049866f9e)

`ScanDy` is a modular and mechanistic computational framework for simulating realistic **scan**paths in **dy**namic real-world scenes. The model aims to simulate gaze behaviour analogous to measured eye-tracking data. While the model was developt based on eye-tracking data of young and healthy adults, this work tried to train the model also on PD patients, as well as age-matched humans. Goal is to find differences in Scanpathmodulation caused by age and/or dissease. 

## Dataset

The used and preprocessed data can be found in: 

Describtion of preprocessing can be found in: 

## Result structure
Each folder includes the evol-file as well as a simulation on the top_3 parameter fits on test and traindata. 
Additional analysing.ipynb creates afterwards a PDF summary also safed in the corresponding folder. 
The pdf summary includes the model fit parameters and scores, the BDIR ratio comparision between model and humand data (train and test), as well as the BDIR timecourse comparision (tain and test) and a more detailed comparision of each BDIR category alone over time against the train data results (including RMSE). Additional for one scene (choosen as it has not so many objects make it easier to see something but can be changed) the percentage of each object (and background) are shown. Attention at this point, the naming is different in human data and model data (B and Ground is background and then no matter if the object has a name or is just named object the number are corresponding to the same object, e.g. plank_1 and Object_1 refers to the same objcet). 

The evolution plot showing development of fit scores for each fit score over generation that is really helpfull to evaluate if the Algorithm settings is okay, isn't saved in the PDF but can be generated in analysing.ipynb as well. 

## More information

Read my report for more details about my findings and work. 

## Examples for ScanDy (from (Roth et al., 2023))

The original examples from Nico can be still find in..

* [Example 1](examples/ex1_scanpath_sgl_video.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex1_scanpath_sgl_video.ipynb): Scanpath simulation and visualization for a single video
* [Example 2](examples/ex2_model_comparison.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex2_model_comparison.ipynb): Evolutionary optimization of model parameters
* [Example 3](examples/ex3_model_extension.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex3_model_extension.ipynb): Extending on existing models: Location-based model with object-based sensitivity
 
### The original Paper introducing ScanDy

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
