See https://drive.google.com/drive/folders/1xO-DYfnMF9cYDJLareTQDKJNokYqukM1?usp=sharing for the full data. Please see description from the below:

## Mobility and Simulation Data
- pairs_full_12.npy: the data generated from initial safegraph data. this is a dict that counts the number of visitors from a CBG to a POI 
- bipartite_weight_12.npy: bipartite graph generated from pairs_full_12.npy; adjusted by areas and dwelll time
- dict_param_all_12.npy: the input file for US simulation
- dict_param_synthetic_[float].npy: input files for synthetic network simulation. the float is the lambda
- state_dict_param/dict_param_*.npy state-level dict_params, numbers are state codes
- poi2area.npy: area for each POI
- poi2dwell_corrects_total_12.npy: median dwell time for each POI

## Census and Vaccination data
- census_cbg_with_predicted_hesitancy_vaccincation.csv: CBG level census data
- cc-est2019-agesex-48.csv: Texas census data
- COVID-19_Vaccinations_in_the_United_States_County.csv: US vaccination data
- texas-07-01-2021.xlsx: Texas vaccination data

## Intermediate Results
- vac_inferred_lvm.csv: Inferred CBG level vaccination rate 
- campaign.npz: the proposed vaccination campaign 

