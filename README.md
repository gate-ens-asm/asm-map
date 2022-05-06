# asm-map

## Project description
This script is dedicated to the detection of Artisanal and Small-Scale Mines (ASM) in Africa.

In practice, this project makes use of satellite hyperspectral imagery, combined with Deep Learning model training, in order to detect and map the ASM activities.
The images we use are pre-processed Sentinel-2 images freely provided by the European Space Agency (ESA).
The kind of models used to perform predictions are Convolutionnal Neural Networks, such as U-net.

At the moment, this repository only contains the 'prediction' and 'evaluation' parts of the pipeline, as the project is not over yet.
A pre-trained U-net model is made available, accompanied by a set of 26 images - on which it was not trained on - to evaluate its performances.
The metrics computed to quantify the model's performances are: Confusion Matrix, Precision, Recall, F(beta = 0.5) score, ROC curve and corresponding AUC.
The provided model was trained on 81 images of ~ 15km x 15km from the Sub-Tropical West Africa, divided into training and validation datasets.
The provided 26 images also covers ~ 15km x 15km each, and are also originated from the Sub-Tropical West Africa while representing its diversity, hence constituing our test dataset. 


## License
This repository is open source under the Creative Commons Attribution (CC BY) 4.0 license.
It means that its content is freely available, and any third party is permitted to access, download, copy, distribute, and use these materials in any way, even commercially, with proper attribution.


## Repository content
TBD
