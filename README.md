# asm-map

## Project description
This script is dedicated to the **detection of Artisanal and Small-Scale Mines (ASM) in Africa**.

In practice, this project makes use of satellite hyperspectral imagery, combined with Deep Learning model training, in order to detect and map the ASM activities.
The images we use are pre-processed Sentinel-2 images freely provided by the European Space Agency (ESA).
The kind of models used to perform predictions are Convolutionnal Neural Networks, such as U-net.

At the moment, this repository only contains the 'prediction' and 'evaluation' parts of the pipeline, as the project is not over yet.
A pre-trained U-net model is made available, accompanied by a set of 26 images to evaluate its performances:

- The provided model was trained on 81 images of ~ 15km x 15km from the Sub-Tropical West Africa, divided into training (75%) and validation (25%) datasets.
- The provided independent 26 images also cover ~ 15km x 15km each, and are also originated from the Sub-Tropical West Africa region. As the model was not trained on these images, and as they are spread over the region while representing its landscape diversity (wild rainforest, shrublands, cities and urban areas, crops and fiels, coasts, mangroves etc.), this set hence constitute our test dataset used to evaluate the model.
- The metrics computed to quantify the model's performances are: Confusion Matrix, Precision, Recall, F(beta = 0.5) score, ROC curve and corresponding AUC.


## License
This repository is open source under the Creative Commons Attribution (CC BY) 4.0 license.
It means that its content is freely available, and any third party is permitted to access, download, copy, distribute, and use these materials in any way, even commercially, with proper attribution.


## Environment
An environment file named 'asm-map-env.yaml' is provided within the asm-map repository's main folder.

It contains a list of all the necessary packages and their version to properly run the scripts. It is strongly recommended to use it as some parts of the asm-map scripts may only work when using the right package's version (especially tensorflow). Also, note that the python version used is 3.8.

Consequently, one should create a dedicated environment easily by using this file.
For instance, by using the **Anaconda** environment manager, one can run in the CLI (from the asm-map repository):
```
$ conda env create -n asm-map-env --file=./asm-map-env-test.yaml
```

## Data access

While the lower-weigth data is already contained into the repository, some of the needed data was to heavy to be stored on GitHub and is hence stored on Dropbox.
Are stored there: the U-net model, the raster test dataset's images (both images to be predicted and their corresponding ground truth ASM shapes) and the African biomes shapefile we make use of.
One can access it through this link : [DATA](https://www.dropbox.com/sh/qdyw5gk3sid33ny/AACWHA0lnjeuh9Ya3mYsPuSGa?dl=0)

**In order to run the prediction / evaluation scripts properly, one must download and copy from this link:**
- the **images** folder (with its content) should be downloaded, unzipped and copied into the *'asm-map/data/test-images/Sub-Tropical_West_Africa'* folder **(~2.6 GB)**;
- the **ground-truth** folder (with its content) should also be downloaded, unzipped and copied into the *'asm-map/data/test-images/Sub-Tropical_West_Africa'* folder **(~55 MB)**;
- the **africa_biomes_wwf.xxx** 4 files should be downloaded (unzipped) and copied all together into the *'asm-map/data/shapefiles'* folder **(~62 MB)**;
- the **model.h5** file should be downloaded and copied into the *'asm-map/data/models/Unet_2021-05-16/data'* folder **(~372 MB)**.


## Repository content

- **CONFIG FOLDER**:
  - It contains the **YAML configuration file needed to execute the main script**: *'config.yaml'*. Some parameters are pre-filled, but **it needs to be modified** to match your local paths and your expectations. Note that the pre-filled parameters should usually not be modified, except from the parts between brackets (i.e. *'[/home/user]'*), which must be removed/modified before to launch the script (otherwise an Exception will be raised). It can be modified from any text or CLI editor, for instance with common nano editor, run (from main *'asm-map'* folder):
  ```
  $ sudo apt install nano      # Optional: If nano package not already installed on your machine
  $ nano config/config.yaml    # Edit file, save with 'Ctrl+S' then exit with 'Ctrl+X'
  ```
  - It also contains an example configuration file (*'example_config.yaml'*) to help you fill out the missing parameters and understand their meaning. For more information on the required parameters, one should read directly the functions' documentations into the *'src/predict.py'* and *'src/evaluate.py'* scripts, which are more documented.


- **DATA FOLDER**:
  - It contains 3 sub-folders: *'models'*, *'shapefiles'* and *'test-images'*.
  - The *'models'* sub-folder contains all the model-related files, and if one wants to try another model that the one we provide, it should be stored here. Note that the *'model.h5'* file should have been imported from the Dropbox repository during the previous step into the recommended sub-folder.
  - The *'shapefiles'* sub-folder contains the shapefiles used during the evaluation process. Note that the *''africa_biomes_wwf.xx* files should have been imported from the Dropbox repository during the previous step into this sub-folder.
  - The *'test-images'* sub-folder contains all the images-to-be-predicted related files, and particularly it aims to contain our test dataset into its *'Sub-Tropical_West_Africa'* sub-folder, as specified in the previous step. If one wants to try another set of images than the one we provide, it should be stored here. But please note that the images we use are derived from Sentinel-2 satellite products, which have been pre-processed and reworked a lot, and that this pre-processing has to be perfectly reproduced so that our model could process the input images. This pipeline is quite trigerring and time-consuming, but still it is doable if desired, please refer to the linked documentation to do so.


- **SRC FOLDER**:
  - It contains all the scripts needed to either run a prediction or an evaluation.
  - The main scripts are directly stored into the *'src'* folder, whereas the useful sub-scripts are stored into the *'utils'* subfolder with appropriate namings.


## Launching the scripts

First:
- 1- **Create the needed *asm-map-env* environment** (cf 'Envrionment' section above);
- 2- **Import the needed missing data** from Dropbox (cf 'Data access' section above);
- 3- **Update the configuration file** (cf 'Repository content' section / 'Config folder' sub-section above).

Then, run in the CLI (from the *'asm-map'* folder):
```
$ conda activate asm-map-env
$ python src/main.py
```
