# gingko-abdev-competition-2025-polyreactivity-open-source-submission

## Description of contents
* data: folder containing all the data used for model training. I used only publicly available data / features provided by Gingko
* utils: folder containing accessory classes and functions for feature generation and handling, as well as model runs
* submitted_11172025_pr_cho: folder containing the CSV files and fitted models that were submitted for the competition
* outputs: folder where all outputs from the FINAL_pr_cho_model pipeline notebooks are stored   

PYTHON NOTEBOOKS
* FINAL_pr_cho_model_pipeline.ipynb:  this notebook contains the complete pipeline for training and evaluating the model as well as creating the final submission files. You should be able to just run this notebook to get a reproduction of the competition submission in the outputs folder. The "data_path” variable(s) shouldn’t need to be updated if you’re running within the same location but it is an option in case its needed.   
* pr_cho_data_exploration.ipynb: accessory notebook containing analysis used to support some of the decisions made in the final model pipeline (for e.g. not imputing missing features etc.) 
