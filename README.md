# Airline-Delay-Prediction-XgBoost-Level-Hard-Difficulty-MSML610
Airline Delay Prediction project (hard difficulty) for MSML 610

<br> Here is a link for the instructions (so I can access it easily instead of digging for it): https://github.com/gpsaggese-org/umd_classes/blob/master/class_project/instructions/README.md

<br> Here is a link for the specific project I have (level hard on the bottom): https://github.com/gpsaggese-org/umd_classes/blob/master/class_project/MSML610/Fall2025/project_descriptions/xgboost_Project_Description.md 

## Note: 
If you clone my repository, all you really have to do see the visuals and all my work is run 05_running_app.ipynb. There, you will see the summaries of my XgBoost as well as the bonus points of comparing it with Catboost alongside GBM. Likewise, you can see the indivdual plots and visuals for each of the 3 models and performance metrics such as confusion matrices, lost plot on train/val, ROC/AUC, etc. Of course, there was a lot that went into cleaning and preparing the data so we have "good data in, good data out". Similarly, you can see the backbone and deep underlying of my work that went into modeling and tuning the models - specifically XGBoost. Lukcily, you won't have to suffer the dozens of hours that it took me to run the models and tune them since you can clone it and see the outputs in the folders and .ipynb files.

## Reproduction On Command Line (Once you have cloned my repository:

1) Create env

conda create -n airline-delay-prediction python=3.10 -y

conda activate airline-delay-prediction

pip install -r requirements.txt

(CatBoost sometimes needs OpenMP on Mac; LightGBM wheel covers most setups)

2) Produce features & train (XGB tuned + baselines):

Chose 1 of the ways to run:

a. If you wanna run .py files:

python -m src.spark_etl          
  or: notebooks/01_spark_etl_and_features.ipynb

python -m src.train_xgb           

python -m src.train_baselines    


b. Or if you wanna run .ipynb files:

notebooks/01_spark_etl_and_features.ipynb

notebooks/03_train_evaluate_model.ipynb

notebooks/04_tuning_models_ex.ipynb

3) Launch app

streamlit run src/app.py

Or:

notebooks/05_running_app.ipynb
