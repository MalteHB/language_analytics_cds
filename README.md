# Language Analytics - Spring 2021

This repository contains all of the code and data related to the Spring 2021 module _Language Analytics_ as part of the bachelor's tilvalg in [Cultural Data Science](https://bachelor.au.dk/en/supplementary-subject/culturaldatascience/) at Aarhus University.

This repository is in active development, with new material being pushed on a weekly basis. 

## Technicalities

To run and use the Python files located in the `src/` folder, I recommend installing [Anaconda](https://docs.anaconda.com/anaconda/install/) and using `conda` to administrate your environments. 

To create an environment capable of running the `.py`files in this repo create run the following code in a terminal:

```bash
# Create conda env:
conda create -n cds python=3.8

# Activate conda env:
conda activate cds

# Install requirements
pip install -r requirements.txt

# Conda install packages
conda install opencv -y
conda install ipykernel -y
```


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

