
# networks for spatial COVID-19 vaccination heterogeneity

This is the replication codes and data for "Mobility network reveals the impact of spatial vaccination heterogeneity on COVID-19"

Preprint: https://www.medrxiv.org/content/10.1101/2021.10.26.21265488v1

Contact: Yuan Yuan, Purdue University, yuanyuan at purdue dot edu

## Overview
We store all Figure 1 related files in the **gephi** folder.

For all codes for other analyes (including processing safegraph data, generate dict_param files for the input of simulation, deep learning, and targeting algorithms), we store in the **code** folder. 

For all **data** and intermediate **results**, we temporally store in the google drive. Before publication we will release them on Zenodo; the reason for the wait is because Zenodo does not allow further additions or modifications of files.

## Dependencies and softwares/programming language

Python 3.8.3 (including packages - numpy, pandas, seaborn, matplotlib, scipy, sklearn, torch)

Gephi 0.92

## gephi

The folder "gephi" contains all data and code that generate the Figure visualization.

gephi/county_construction.ipynb is the python notebook that processes the data.

gephi/node_US_county.csv and edge_US_county.csv are the files used by gephi to generate visualization.

gephi/US.gephi is the Gephi input file; Figure 1 is produced in this way.

gephi/viz_simulation.ipynb: code that generates simple version of synthetic networks for Fig S1

Please download the Gephi software from https://gephi.org/users/download/ and then open US.gephi with Gephi.

## code

### preprocessing


code/process_safegraph.ipynb is the file that processes the initial safegraph data. Please refer to https://www.safegraph.com/covid-19-data-consortium for the COVID-19 related data. It mainly outputs data/pairs_full_12.npy.

code/preprocess.ipynb is the file that further generates input files for the downstream simulation tasks.

### synthetic network analysis

code/generate_synthetic.ipynb is the file that generates the input for simulations on synthetic networks. It also outputs the files shown in the main text.

code/disease_model.py is based on the code derived from https://covid-mobility.stanford.edu// We further incorporate the consideration of vaccination rates.

code/US_simulation-synthetic.py is the simulation code for synthetic networks.

You can run the code in the following way

    python US_simulation-synthetic.py  --vc=-1 --num_hours=720 --p_sick_at_t0=0.0001 --poi_psi=0.1  --home_beta=0.00 --state=synthetic_0.750000 --enable=0

 - vc: a float that fixes the global vaccination rate. -1 means use the average from the input file
 - num_hours: number of hours to simulate
 - p_sick_at_t0: % of exposed people at time 0
 - poi_psi: cross CBG transmission parameter (see https://covid-mobility.stanford.edu//)
 - home_beta: within CBG transmission parameter (see https://covid-mobility.stanford.edu//)
 - state: input file name (files available on google drive)
 - enable: used to run all hypothetic distribution (where enable=0) or a single one (1=original, 2=reverse, 3=exchange, 4=shuffle, 6=order)

### US network analysis

code/generate_US.ipynb is the file that generates the input for simulations on synthetic networks. It also outputs the files shown in the main text.

code/disease_model.py is based on the code derived from https://covid-mobility.stanford.edu// We further incorporate the consideration of vaccination rates.

code/US_simulation-track-12.py is the simulation code for US networks.

You can run the code in the following way:

    python US_simulation-track-12.py  --vc=-1 --num_hours=720 --p_sick_at_t0=0.001 --poi_psi=120000000.0  --home_beta=0.005 --state=all --enable=0 --distribution=original --intervene=4

 - vc: a float that fixes the global vaccination rate. -1 means use the average from the input file
 - num_hours: number of hours to simulate
 - p_sick_at_t0: % of exposed people at time 0
 - poi_psi: cross CBG transmission parameter (see https://covid-mobility.stanford.edu//); we need a much larger value here because our poi_psi / 12 / 12 / 720 equals the parameter in https://covid-mobility.stanford.edu// 
 - home_beta: within CBG transmission parameter (see https://covid-mobility.stanford.edu//)
 - state: input file name (files available on google drive)
 - enable: used to run all hypothetic distribution (where enable=0) or a single one (1=original, 2=reverse, 3=exchange, 4=shuffle, 6=order)
 - distribution: original, exchange, shuffle, order, reverse_within
 - intervene: use when we hope to the targeting result (1=optimal, 2=centrality, 3=low_vax, 4=random, 5=non-targeting)

### For deep learning methods

code/small_area.ipynb is the file where the deep learning models are trained and validated

code/small_area_visualize.ipynb is the file where we visualize the results

### For targeting 

code/campaign.ipynb is the file that implements the campaign algorithms

code/viz_campaign.ipynb is the file where produces the figure

We use https://simplemaps.com/ to visualize the targeted CBGs

## data and results

As GitHub has an upper limit for repository size. Currently we uploaded large files to Google Drive (https://drive.google.com/drive/folders/1xO-DYfnMF9cYDJLareTQDKJNokYqukM1?usp=sharing)

Please download the files from the google drive for replication purposes.

## Cautions

For replication of US mobility network, we expect a server with >400 memory and >10h CPU time for a 30-day simulation. Please try synthetic data or a state-level simuation for the quick demo. 
