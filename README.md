## Objectives

This script was used to run experiments described in _Supervised diagnosis prediction from cortical sulci : toward the discovery of neurodevelopmental biomarkers in mental disorders_, accepted to 21th IEEE ISBI 2024.

## Running experiments

### Installation
To install the package, clone the repository into a folder and then :
``` Shell
cd /path/to/diag_pred_from_sulci
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

The 3 clinical datasets `SCZDataset`, `BDDataset` and `ASDDataset` are derived mostly from public cohorts excepted for 
BIOBD, BSNIP1 and PRAGUE, that are private for clinical research. Here, the phenotyp information of each dataset :

**Dataset** | **# Subjects** | **Age** (avg±std) | **Sex (\%F)** | **# Sites** | **Studies**
| :---:| :---: | :---: | :---: | :---: | :---: |
HC<br>SCZ | 761<br>532 | 33 ± 12<br>34 ± 12 | 51<br>29 | 12 | [BSNIP1](http://b-snip.org), [CANDI](https://www.nitrc.org/projects/candi_share), [CNP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664981/),   PRAGUE, [SCHIZCONNECT](http://schizconnect.org)
HC<br>BD | 695<br>469 | 37 ± 14<br>39 ± 12 | 54<br>57 | 15 | [BIOBD](https://pubmed.ncbi.nlm.nih.gov/29981196/), [BSNIP1](http://b-snip.org), [CANDI](https://www.nitrc.org/projects/candi_share), [CNP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664981/)
HC<br>ASD | 926<br>813 | 16 ± 9<br>16 ± 9 | 25<br>13 | 30 | [ABIDE I](http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) , [ABIDE II](http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html)

To run experiments, you need a `root` folder containing :
- the pickles of train-val-test schemes for each dataset
- the mapping of acquisition sites
- a folder `morphologist` with arrays of skeleton volumes and corresponding participant dataframes of each study

### Experiments

1. Architecture selection : 3 CNN architectures have been tested, see the `architecture` folder
2. Loss selection : BCE and SupCon losses have been compared, see `contrastive_learning` folder for SupCon model
3. Pre-processing selection : Gaussian smoothing pre-processing, see `img_preprocessing` folder
4. XAI : an occlusion method have been applied to understand model decisions, see `saliency_map` folder

## Useful links
Link to the paper : <https://hal.science/hal-04494994>

First version of these scripts are at: <https://github.com/Duplums/SMLvsDL>