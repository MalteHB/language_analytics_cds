# Language Analytics - Spring 2021

This repository contains all of the code and data related to the Spring 2021 module _Language Analytics_ as part of the bachelor's tilvalg in [Cultural Data Science](https://bachelor.au.dk/en/supplementary-subject/culturaldatascience/) at Aarhus University.

This repository is in active development, with new material being pushed on a weekly basis. 

## Technicalities

To run and use the Python files located in the `src/` folder, I recommend installing [Anaconda](https://docs.anaconda.com/anaconda/install/) and using `conda` to administrate your environments. 

To create an environment capable of running the `.py`files in this repo create run the following code in a terminal:

```bash
# Clone the GitHub
git clone https://github.com/MalteHB/language_analytics_cds.git
cd language_analytics_cds 

# Create conda env:
conda create -n cds python=3.8

# Activate conda env:
conda activate cds

# Install requirements
pip install -r requirements.txt

# Download the English language model from spaCy
python -m spacy download en_core_web_sm

```

## Assignment 5 - (Un)Supervised Machine Learning

### Project Description
For the fifth assignment of the Language Analytics course a short research project was tasked. 

Since i have delved into both unsupervised and supervised machine learning prior to attending the course, i chose that i wanted to investigate a possible framework to use for text classification. Specifically i wanted to investigate the capabilities of [HuggingFace's](https://huggingface.co/) [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) class in a token classification setup. Furthermore, i wanted to do it for Danish, since Danish currently stands in a  technological disadvantageous position compared to bigger languages such as English. 

For Danish token classification there is, to my knowledge, only one dataset, namely [DaNE](https://www.aclweb.org/anthology/2020.lrec-1.565/). This dataset follows the CoNLL-2003 annotation scheme and does therefore encompass several token classification tasks. 

The token classification task i chose to test was Named Entity Recognition (NER) and the script can be seen [here](src/HF_token_classification.py). The script is a wrapper for the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) class, and what is clever about the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) class is that is comes with all sorts of auto hyperparameter optimization during training. I, therefore, wanted to see whether it was possible to achieve similar F1-scores when using the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) class and a Danish transformer-based model, [Ælæctra](https://github.com/MalteHB/-l-ctra) class compared to the F1-scores reported in the [Ælæctra repository](https://github.com/MalteHB/-l-ctra). 

Using the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) class I managed, through multiple trainings, to achieve F1-scores ranging between 77 and 80 on the predefined testset from [DaNE](https://www.aclweb.org/anthology/2020.lrec-1.565/). I think that the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) class, therefore, serves as a promising tool to use, when wanting to employ more automated supervised machine learning, and the future for NLP in general seems bright. 

### Script Usage
Start of by cloning this repository and creating a virtual environment with the requirements installed. See the [Technicalities](##Technicalities) section to see how this is done using Anaconda.

After initializing your own environment you can run the train a state-of-the-art Danish NER model from a command-line by using the following command:

```bash
python src/HF_token_classification.py --train
```

Once trained you can predict a sentence of your choice by using the '--s' flag. :

```bash
python src/HF_token_classification.py --s "Ross er en dejlig mand, som kommer fra Skotland, og underviser på Aarhus Universitet."
```

For a list of capabilities use the '--help' flag.
## Repo structure

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| A folder to be used for sample datasets that we use in class.
```notebooks``` | This is where you should save all exploratory and experimental notebooks.
```src``` | For Python scripts developed in class and as part of assignments.
```utils``` | Utility functions that are written by me, and which we'll use in class.

## Course overview and readings

A detailed breakdown of the course structure and the associated readings can be found in the [syllabus](syllabus.md), while the _studieordning_ can be found [here](https://eddiprod.au.dk/EDDI/webservices/DokOrdningService.cfc?method=visGodkendtOrdning&dokOrdningId=15952&sprog=en).

## Acknowledgement

All the credits for the content, the syllabus, the teaching and this Git repository goes to [Ross Deans Kristensen-McLachlan](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html). 

## Contact

For help or further information feel free to connect with me, Malte, on [hjb@kmd.dk](mailto:hjb@kmd.dk?subject=[GitHub]%20Language%20Analytics%20Cultural%20Data%20Science) or any of the following platforms:

[<img align="left" alt="MalteHB | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter]
[<img align="left" alt="MalteHB | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="MalteHB | Instagram" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/instagram.svg" />][instagram]

<br />

</details>

[twitter]: https://twitter.com/malteH_B
[instagram]: https://www.instagram.com/maltemusen/
[linkedin]: https://www.linkedin.com/in/malte-h%C3%B8jmark-bertelsen-9a618017b/

