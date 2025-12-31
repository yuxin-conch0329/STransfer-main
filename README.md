# STransfer-main
STransfer: A Transfer Learning-Enhanced Graph  Convolutional Network for Clustering Spatial  Transcriptomics Data
# STransfer reproducibility

This repository offers implementation scripts supporting the experimental procedures outlined in the publication "STransfer: A Transfer Learning-Enhanced Graph Convolutional Network for Clustering Spatial Transcriptomics Data".
<img width="1000" height="800" alt="Fig1" src="https://github.com/user-attachments/assets/101a6b88-675f-4715-8ec7-c3b1ea1144a4" />




# General Flow

It is recommended the user proceeds as follows.

1. Clone this repository to their local machine.
2. Download the data.
3. Install necessary packages.
4. Run STransfer.

## Directory

- `Figures`stores the flow picture and five datasets pictures
- `STransfer` stores the main script of STransfer
- `test` contains the testing script of STransfer on the datasets in the manuscript

## Data source and reference

All datasets used in our study were obtained from previously published sources. The DLPFC dataset is publicly available through the spatialLIBD project. The HPR and
mPFC datasets correspond to Data 25–29 and Data 31–33, respectively, and can be accessed from the SDMBench repository. The CRC dataset is available for download from the CRC website. The OC dataset can be obtained from the Spatch project.

## Install necessary packages

Recomended installation procedure is as follows.

1. Install [Anaconda](https://www.anaconda.com/products/individual) if you do not already have it, so that you can access conda commands in terminal.

2. Create a conda environment, and then activate it as follows in terminal. The code here requires the versions of the packages specified in the `requirements` file. 

   ```
   $ conda env create -n STransfer
   ```

### Run ADDA

We recommend starting with the reproduction of the DLPFC dataset. The ADDA.py file is placed under the STransfer folder. This file implements the Adversarial Domain Adaptation (ADDA) framework used for training on both the source and target domains, with DLPFC used here as an example.

`modules.py`
Defines the core neural network components, including the encoders, classifier, and discriminator used in the model.

`params.py`
Stores all hyperparameter configurations and handles command-line argument parsing.

`utils.py`
Includes utility functions such as metric computation and graph construction.

`train.py`
The main training script that integrates data loading, model training, and evaluation loops.

`HPR.py`
Contains data processing and preprocessing methods for the MERFISH dataset from the mouse hypothalamic preoptic region.

`mPFC.py`
Contains code for processing the STARmap dataset of the medial prefrontal cortex (mPFC).

`CRC.py`
Contains code for processing the dataset of the Colorectal Cancer Tumors (CRC).

`OC.py`
Contains code for processing the Ovarian Cancer Datasets Across Multiple Platforms(OC).
