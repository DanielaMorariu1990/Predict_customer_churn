# Predict Customer Churn

This project is part of the nanodegree **Machine Learning DevOps Engineer** offered by Udacity.

## Project Description

The projet predicts customer churn using a logistic Regression (lr) and Random Forest Classifier (rf).
The purpose of the project is to use clean code principles.
Inside the project I have experimented with the libary MlFlow (which I found quite user friendly) and with the testing library pytest, particularly with fixtures in pytest.

## Files and data description

**data** folder contains the churn customer data in a csv file. This data is used for training.

**images** folder contains two subfolders:

- eda: Contains plots from the EDA, created by the method ChurnLibrary.perform_eda
- results: Contains plots summarizing results, after training the 2 models (lr and rf). These plots are created by multiple class methods: ChurnLibrary.train_model (creates roc curve plot), ChurnLibrary.classification_report_image (creates the two classification reports) and ChurnLibrary.classification_report_image.feature_importance_plot (creates the feature importance plot for rf).

**logs** folder contains two log files: one from running the actual library (training the models) and one contains the results of the testing framework.

**lr_model** contains the trained lr model, including model.pkl file. These folder is created using the mlflow framework.

**mlruns** contains the ovreview of the experiments condacted by running the churn_libabry.py, using mlflow. This is generated by mlflow out of the box.

**rf_model** contains the trained rf model, including the model.pkl file.

**churn_library.py** containes the class ChurnLibrary, which is a module that performs the import of teh data, the EDa, the training and the saving of the model using Mlflow, and finaly the generating and storing of the multiple results plot.

**churn_script_logging_and_test.py** contains the tests for the class ChurnLibrary and it's most important methods. For this purpose I have used pytest framework and made use of fixtures.

**constants_pytest.py** I exported the constants in a separte file, which I use for the _churn_library.py_ , as well as _churn_script_logging_and_test.py_.

**helper_functions.py** contains the helper functions for the module ChurnLibrary. These functions haven't been includede in the test. Might be a future improvement.

**requirements.txt** contains the requirements of running this project.

## Running Files

In order to run the files, you need to first install all the requirements:

```
pip install -r requirements.txt

```

In order to run the **churn_library.py**, you can do so from the command line after installing needed packages. These will produce also logs in the log file _churn_libabry_running.log_.

```
python churn_library.py

```

In order to run the **churn_script_logging_and_test.py**, you can do so from the command line after installing needed packages. These will produce also logs in the log file _churn_library_test_.

```
pytest churn_script_logging_and_tests.py

```
