## Objectives

This script was used to run experiments described in _Supervised diagnosis prediction from cortical sulci : toward the discovery of neurodevelopmental biomarkers in mental disorders_, accepted to 21th IEEE ISBI 2024.

## Running experiments

### Installation
To install the package, clone the repository into a folder and then :
``` Shell
cd /path/to/folder
pip install -e .
```

### Launch model trainings

To launch model trainings, you need to launch the python script `main.py` in the `dl_training` folder.
All the parameters to be passed into argument are explained in the script.
``` Shell
python3 dl_training/main.py --args
# if you need details about parameters
python3 dl_training/main.py --help
```

### Datasets

The 3 clinical datasets `SCZDataset`, `BipolarDataset` and `ASDDataset` are derived mostly from public cohorts excepted for 
BIOBD, BSNIP and PRAGUE, that are private for clinical research. These 3 datasets are based on the following sources.

**Source**  | **Disease** | **# Subjects** | **Age (avg±std)** | **Sex (\%F)** | **# Sites**
| :---:| :---: | :---: | :---: | :---: | :---: |
[BSNIP](http://b-snip.org)  | Control<br>Schizophrenia<br>Bipolard Disorder | 198<br>190<br>116 | 32 ± 12<br>34 ± 12<br>37 ± 12 | 58<br>30<br>66 | 5
[SCHIZCONNECT](http://schizconnect.org)  | Control<br>Schizophrenia | 275<br>329 | 34 ± 12<br>32 ± 13 | 28<br>47 | 4
PRAGUE  | Control | 90 | 26 ± 7 | 55 | 1
[BIOBD](https://pubmed.ncbi.nlm.nih.gov/29981196/) | Control<br>Bipolar Disorder | 306<br>356 | 40 ± 12<br>40 ± 13 | 55 | 8
[CANDI](https://www.nitrc.org/projects/candi_share) | Control<br>Schizophrenia | 25<br>20 | 10 ± 3<br>13 ± 3 | 41<br>45 | 1
[CNP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664981/) | Control<br>Schizophrenia<br>Bipolar Disorder | 123<br>50<br>49 | 31 ± 9<br>36 ± 9<br>35 ± 9| 47<br>24<br>43 | 1 
ABIDE 1 | Control<br>Autism Spectrum Disorder | 404<br>372 | 18 ± 9<br>18 ± 9 | 17<br>12 | 16
ABIDE 2 | Control<br>Autism Spectrum Disorder | 543<br>459<br> | 15 ± 9<br>15 ± 9 | 32<br>14 | 19

To run experiments, you need a `root` folder containing :
- the pickles of train-val-test schemes for each dataset
- the mapping of acquisition sites
- the mapping of disease
- a folder `morphologist` with arrays of skeleton volumes and participant dataframes of each study

### Experiments

1. Architecture selection : 3 CNN architectures have been tested, see the `architecture` folder
2. Loss selection : BCE and SupCon losses have been compared, see `contrastive_learning` folder for SupCon model
3. Pre-processing selection : Gaussian smoothing pre-processing, see `preprocessing` folder
4. XAI : an occlusion method have been applied to understand model decisions, see `saliency_map` folder

## Useful links
Link to the paper : <https://hal.science/hal-04494994>

First version of these scripts are at: <https://github.com/Duplums/SMLvsDL>