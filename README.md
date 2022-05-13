# asm-map

## Project description
This script is dedicated to the detection of Artisanal and Small-Scale Mines (ASM) in Africa.

In practice, this project makes use of satellite hyperspectral imagery, combined with Deep Learning model training, in order to detect and map the ASM activities.
The images we use are pre-processed Sentinel-2 images freely provided by the European Space Agency (ESA).
The kind of models used to perform predictions are Convolutionnal Neural Networks, such as U-net.

At the moment, this repository only contains the 'prediction' and 'evaluation' parts of the pipeline, as the project is not over yet.
A pre-trained U-net model is made available, accompanied by a set of 26 images - on which it was not trained on, hence constituting our test dataset - to evaluate its performances.
The metrics computed to quantify the model's performances are: Confusion Matrix, Precision, Recall, F(beta = 0.5) score, ROC curve and corresponding AUC.
The provided model was trained on 81 images of ~ 15km x 15km from the Sub-Tropical West Africa, divided into training and validation datasets.
The provided 26 images also covers ~ 15km x 15km each, and are also originated from the Sub-Tropical West Africa while representing its diversity, hence constituing our test dataset. 


## License
This repository is open source under the Creative Commons Attribution (CC BY) 4.0 license.
It means that its content is freely available, and any third party is permitted to access, download, copy, distribute, and use these materials in any way, even commercially, with proper attribution.


## Environment
An environment file named 'asm-map-env.yaml' is provided within the asm-map repository's main folder.

It contains a list of all the necessary packages and their version to properly run the scripts. It is strongly recommended to use it as some parts of the asm-map scripts may only work when using the right package's version. Also, note that the python version used is 3.8.

Consequently, one should create a dedicated environment easily by using this file.
For instance, by using the **anaconda** environment manager, one can run in the CLI (from the asm-map repository):
```
$ conda env create -n asm-map-env --file=./asm-map-env-test.yaml
```

## Data access

While the lower-weigth data is already contained into the repository, some of the needed data was to heavy to be stored on GitHub and is hence stored on Dropbox.
Are stored there: the U-net model, the raster test dataset's images (both images to be predicted and their corresponding ground truth ASM shapes) and the African biomes shapefile we make use of.
One can access it through this link : [DATA](https://www.dropbox.com/sh/qdyw5gk3sid33ny/AACWHA0lnjeuh9Ya3mYsPuSGa?dl=0)

**In order to run the prediction / evaluation scripts properly, one must download and copy from this link:**
- the **images** folder (with its content) should be copied into the *'asm-map/data/test-images/Sub-Tropical_West_Africa'* folder;
- the **ground-truth** folder (with its content) should also be copied into the *'asm-map/data/test-images/'* folder;
- the **africa_biomes_wwf.xxx** 4 files should be copied together into the *'asm-map/data/shapefiles'* folder;
- the **model.h5** file should be copied into the *'asm-map/data/models/Unet_2021-05-16/data'* folder.


## Repository content

- **CONFIG FOLDER**:
  - It contains the YAML configuration file needed to execute the main script: *'config.yaml'*. Some parameters are pre-filled but **it needs to be modified** to match your local paths and your expectations. Note that the pre-filled parameters should not be modified except from the parts between brackets (i.e. *'[/home/user]'*).
  - It also contains an example configuration file (*'example_config.yaml'*) to help you fill out the missing parameters.
  
- **DATA FOLDER**:
  - It contains 3 sub-folders: *'models'*, *'shapefiles'* and *'test-images'*.
  - The *'models'* sub-folder contains all the model-related files, and if one wants to try another model that the one we provide, it should be stored here. Note that the *'model.h5'* file should have been imported from the Dropbox repository during the previous step into the recommended sub-folders.
  - The *'shapefiles'* sub-folder contains the shapefiles used during the evaluation process. Note that the *''africa_biomes_wwf.xx* files should have been imported from the Dropbox repository during the previous step into this sub-folder.
  - The *'test-images'* sub-folder contains all the images-to-be-predicted related files, and particularly it aims to contain our test dataset into its *'Sub-Tropical_West_Africa'* sub-folder, as specified in the previous step. If one wants to try another set of images that test dataset we provide, it should be stored here (but please note that the images are derived from Sentinel-2 satellite images products which have been pre-processed and reworked a lot, and that it has to be perfectly reproduced so that our model could process the input images. This pipeline is quite trigerring and time-consuming, but still it is doable if desired, please refer to the linked documentaiton to do so).

- **SRC FOLDER**:
  - It contains all the scripts to either run a prediction or an evaluation.
  - The main scripts are directly stored into the *'src'* folder, whereas the useful sub-scripts are stored into the *'utils'* subfolder with appropriate namings.


## Launching the scripts

First:
- 1- Create the needed *asm-map-env* environment (cf 'Envrionment' section above);
- 2- Import the needed missing data from Dropbox (cf 'Data access' section above);
- 3- Update the configuration file (cf 'Repository content' section / 'Config folder' sub-section above).

Then, run in the CLI from the asm-map folder:
```
$ conda activate asm-map-env
$ python src/main.py
```




