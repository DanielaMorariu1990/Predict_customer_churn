"""
This file containes the libabry for analysing and modeling customer churning.

Author: Daniela Bielz
Date: 20.10.2023
"""

import os
import shutil
import logging
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    plot_roc_curve,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


sns.set()

os.environ["QT_QPA_PLATFORM"] = "offscreen"

logging.basicConfig(
    filename="./logs/churn_library_running.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
logger_cls = logging.getLogger("ChurnLibraryClass")


def check_if_directory_exits_and_create(pth, logger):
    """
    Helper function to check if a directory exists and if not to create it.
    input:
        pth: directory path
    output:
        None

    """
    if not os.path.exists(pth):
        os.makedirs(pth)
        logger.info(f"Directory does not exist! Created directory: {pth}.")
    else:
        logger.info(f"Directory {pth} already exists. Check completed, moving on...")
        pass


def remove_local_model_directory(pth, logger):
    """
    Helper function that removed model directories where we store mlflow model localy.
    input:
        pth: array of directories names
    output:
        None
    """
    for model_pth in pth:
        if os.path.exists(model_pth):
            try:
                shutil.rmtree(model_pth)
                logger.info(
                    f"SUCCESS: Directory {model_pth} removed successfully. We need to delete local model directory for Mlflow process."
                )

            except OSError as o:
                logger.exception(f"Error, {o.strerror}: {model_pth}")
        else:
            logger.info(
                f"SUCCESS: Directory {model_pth} dose not exist. No need to remove it."
            )


def eval_metrics(actual, pred):
    """
    Helper function for calculating metrics for classification problems.

    inputs:
        actual: array, containing actual observations of response variable
        pred: array, containg predicted values for response variable

    output:
        metrics: precision, recall, f1

    """
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return precision, recall, f1


def save_roc_curve_results(model_arr, x_test, y_test, pth, logger):
    """
    Helper function to save the ROC curve for an input of model arrays. Currently implemented only for a model array of length max 2.

    inputs:
        model_arr: array, sklearn models
        x_test: array, x-values for test data set
        y_test: array, y-values for test data set
        pth: path to storage
        logger: logger object

    output:
        None

    """
    plt.figure(figsize=(15, 8))
    if len(model_arr) == 2:
        lrc_plot = plot_roc_curve(model_arr[0], x_test, y_test)
        ax = plt.gca()
        rfc_disp = plot_roc_curve(model_arr[1], x_test, y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
    elif len(model_arr) == 1:
        rfc_disp = plot_roc_curve(model_arr[0], x_test, y_test, ax=ax, alpha=0.8)
    else:
        msg = "Not implemented!!Currently only supporting array length 1 or 2."
        logger.exception(msg)
        raise Exception(msg)
    plt.savefig(f"{pth}/roc_curve_result.png")
    plt.close()


def save_classification_report(
    description,
    actual_test,
    predicted_test,
    actual_train,
    predicted_train,
    pth,
    figsize=(5, 5),
):
    """
    Helper function to save classification reports for diffrent models (test and train).
    input:
        description: str, model descriprion
        actual_test: array, observed response variable test data set
        predicted_test: array, predicted values on test data set
        actual_train:  array, observed response variable train data set
        predicted_train: array, predicted values on train data set
        pth: path to storage
        figsize: figure size for matplotlib plot

    output:
        None

    """
    plt.rc("figure", figsize=figsize)
    plt.text(
        0.01,
        1.25,
        str(f"{description} Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(actual_test, predicted_test)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str(f"{description} Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(actual_train, predicted_train)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(f"{pth}/{description}_results.png")
    plt.close()


class ChurnLibrary:
    """
    The class supports the modeling of churning customers.
    It performs EDA, feature engineering, training and saving results
    (incl. model artifacts and plots)

    """

    def __init__(self, pth, category_lst, logger) -> None:
        self.path = pth
        self.category_list = category_lst
        self.logger = logger
        self.data_frame = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.y_train_preds_rf = None
        self.y_test_preds_rf = None
        self.y_train_preds_lr = None
        self.y_test_preds_lr = None
        self.mlflow_run_id = None

    def import_data(self):
        """
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                self.data_frame: pandas dataframe
        """
        self.data_frame = pd.read_csv(self.path)
        self.logger.info("SUCCESS: Completed reading in data.")

        return self.data_frame

    def perform_eda(
        self,
        dict_columns_to_plot_type,
        path_to_storage="./images/eda",
        figure_size=(20, 10),
        include_heatmap=True,
    ):
        """perform eda on self.data_frame and save figures to images folder
        input:
                self.data_frame: pandas dataframe
                dict_columns_to_plot_type: dictionary,
                 keys: column names of self.data_frame,
                 values: vector;
                        plot_types:normalized_histogram, count_histogram, density_plot as plot types
                        plot_names: names to save the plots

                path_to_storage: string, path to storage
                figure_size: tuple, figure size of plot, defaults to (20, 10)
                include_heatmap: binary, should heatmap be plotted and saved, defaults to True

        output:
                None
        """
        check_if_directory_exits_and_create(pth=path_to_storage, logger=self.logger)

        for column_name, plot_type_name_vector in dict_columns_to_plot_type.items():
            plt.figure(figsize=figure_size)
            if plot_type_name_vector[0] == "normalized_histogram":
                sns_plot = sns.histplot(self.data_frame[column_name], stat="proportion")
            elif plot_type_name_vector[0] == "count_histogram":
                sns_plot = sns.histplot(self.data_frame[column_name], stat="count")
            elif plot_type_name_vector[0] == "density_plot":
                sns_plot = sns.histplot(
                    self.data_frame[column_name], stat="density", kde=True
                )
            sns_plot.set_title(f"{column_name}")
            fig = sns_plot.get_figure()
            fig.savefig(f"{path_to_storage}/{ plot_type_name_vector[1]}")
            plt.close()

        if include_heatmap:
            plt.figure(figsize=figure_size)
            corr_plot = sns.heatmap(
                self.data_frame.corr(), annot=False, cmap="Dark2_r", linewidths=2
            )
            corr_plot.set_title("Correlation Plot (Heatmap)")
            fig = corr_plot.get_figure()
            fig.savefig(f"{path_to_storage}/heatmap.png")
            plt.close()

        self.logger.info(f"SUCCESS: Saved EDA plots to {path_to_storage}.")

    def encoder_helper(self, response):
        """
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                self.data_frame: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could
                be used for naming variables or index y column]

        output:
                self.data_frame: pandas dataframe with new columns for
        """
        self.data_frame["Churn"] = self.data_frame["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        if len(self.category_list) != len(response):
            message = f"Length of names of the columns and columns to be computed dose not match:\
                {len(response)} and {len(self.category_list)}"
            self.logger.exception(message)
            raise Exception(message)
        for col, name_col in zip(self.category_list, response):
            col_lst = []
            grouped_df = self.data_frame.groupby(col).mean()["Churn"]
            for val in self.data_frame[col]:
                col_lst.append(grouped_df.loc[val])

            self.data_frame[name_col] = col_lst

        self.logger.info("SUCEESS: Created one hot encoding for categorical variables.")

        return self.data_frame

    def perform_feature_engineering(self, response):
        """
        input:
                self.data_frame: pandas dataframe
                response: string of response name [optional argument that could
                be used for naming variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        """
        X = self.data_frame[response]
        y = self.data_frame["Churn"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        self.logger.info("SUCEESS: Split data set in test and train data sets.")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(
        self,
        param_grid,
        path_to_images="./images/results",
        path_to_lr_model="lr_model",
        path_to_rf_model="rf_model",
    ):
        """
        train, store model results: images + scores, and store models
        input:
                param_grid: params for grid search for rf model
                path_to_images: path to storing results images
                path_to_lr_model: path to string lr model
                path_to_rf_model: path to storing rf model
        output:
                None
        """
        check_if_directory_exits_and_create(pth=path_to_images, logger=self.logger)
        remove_local_model_directory(
            pth=[path_to_lr_model, path_to_rf_model], logger=self.logger
        )

        with mlflow.start_run() as run:
            self.mlflow_run_id = run.info.run_id
            rfc = RandomForestClassifier(random_state=42)
            lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

            cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
            cv_rfc.fit(self.X_train, self.y_train)
            mlflow.log_param(f"best_params_rf", cv_rfc.best_params_)

            lrc.fit(self.X_train, self.y_train)
            self.logger.info("SUCCESS: Training completed...")

            self.y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
            self.y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)

            self.y_train_preds_lr = lrc.predict(self.X_train)
            self.y_test_preds_lr = lrc.predict(self.X_test)

            self.logger.info(
                "SUCCESS: Competed calculations of predicted values for both test and train."
            )

            for actual, predicted, description in zip(
                [self.y_test, self.y_train],
                [
                    self.y_test_preds_rf,
                    self.y_train_preds_rf,
                    self.y_test_preds_lr,
                    self.y_train_preds_lr,
                ],
                [
                    "Test_Random_Forest",
                    "Train_Random_Forest",
                    "Test_Linear_Regression",
                    "Train_Linear_Regression",
                ],
            ):
                precision, recall, f1 = eval_metrics(actual, predicted)
                self.logger.info(f"{description} results:")
                self.logger.info(f"{description} precision results:{precision} ")
                self.logger.info(f"{description} recall results:{recall} ")
                self.logger.info(f"{description} f1 results:{f1} ")
                mlflow.log_metric(f"{description}_precision_results", precision)
                mlflow.log_metric(f"{description}_recall_results", recall)
                mlflow.log_metric(f"{description}_f1_results", f1)

            save_roc_curve_results(
                model_arr=[cv_rfc.best_estimator_, lrc],
                x_test=self.X_test,
                y_test=self.y_test,
                pth=path_to_images,
                logger=self.logger,
            )
            mlflow.log_artifact(f"{path_to_images}/roc_curve_result.png")
            self.logger.info(
                "SUCCESS: Saved ROC plot locally and as artifact to the run."
            )

            # save best model

            mlflow.sklearn.save_model(
                sk_model=cv_rfc.best_estimator_,
                path=f"./{path_to_rf_model}",
            )
            mlflow.sklearn.log_model(
                sk_model=cv_rfc.best_estimator_,
                artifact_path=path_to_rf_model,
                registered_model_name="rf_model",
            )

            mlflow.sklearn.save_model(
                sk_model=lrc,
                path=f"./{path_to_lr_model}",
            )
            mlflow.sklearn.log_model(
                sk_model=lrc,
                artifact_path=path_to_lr_model,
                registered_model_name="logistic_model",
            )
            self.logger.info(
                "SUCEESS: Trained LR and RF model saved locally to path ./model_lrc and ./model_rf and registerd through mlflow."
            )

    def classification_report_image(
        self,
        path_to_images,
        model_version=3,
        model_path_rf="./rf_model",
        model_path_lr="./lr_model",
    ):
        """
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                path_to_images: path to image results folder
                model_version: int; [optional] represents the last model version registered. Helps us retrieve the last run

        output:
                None
        """
        client = MlflowClient()

        # Deafult to previou run Id if current run Id is not given
        if self.mlflow_run_id is None:
            self.mlflow_run_id = client.get_model_version(
                "rf_model", model_version
            ).run_id
            self.logger.info(
                "Run_id is not defined. Default the run Id, to previous run_id (as indicated by last registered model version)."
            )

        check_if_directory_exits_and_create(pth=path_to_images, logger=self.logger)

        if (self.y_test_preds_rf is None) or (self.y_train_preds_rf is None):
            model_rf = mlflow.sklearn.load_model(f"{model_path_rf}")
            self.y_test_preds_rf = model_rf.predict(self.X_test)
            self.y_train_preds_rf = model_rf.predict(self.X_train)
            self.logger.info("SUCCESS: Calculating predicted values fro RF model.")

        if (self.y_test_preds_lr is None) or (self.y_train_preds_lr is None):
            model_lr = mlflow.sklearn.load_model(f"{model_path_lr}")
            self.y_test_preds_lr = model_lr.predict(self.X_test)
            self.y_train_preds_lr = model_lr.predict(self.X_train)
            self.logger.info("SUCCESS: Calculating predicted values fro LR model.")

        self.logger.info("SUCCESS: Predicted values already calculated.")

        save_classification_report(
            description="Random_Forest",
            actual_test=self.y_test,
            predicted_test=self.y_test_preds_rf,
            actual_train=self.y_train,
            predicted_train=self.y_train_preds_rf,
            pth=path_to_images,
            figsize=(5, 5),
        )
        self.logger.info(
            f"SUCCESS: Saved Random Forest classification report image to specified folder:{path_to_images}"
        )
        client.log_artifact(
            self.mlflow_run_id, f"{path_to_images}/Random_Forest_results.png"
        )
        save_classification_report(
            description="Logistic_Classification",
            actual_test=self.y_test,
            predicted_test=self.y_test_preds_lr,
            actual_train=self.y_train,
            predicted_train=self.y_train_preds_lr,
            pth=path_to_images,
            figsize=(5, 5),
        )
        client.log_artifact(
            self.mlflow_run_id, f"{path_to_images}/Logistic_Classification_results.png"
        )
        self.logger.info(
            f"SUCCESS: Saved Logistic Model classification report image to specified folder:{path_to_images}"
        )

    def feature_importance_plot(
        self, model_path="/rf_model", path_to_images="./images/results", model_version=3
    ):
        """
        creates and stores the feature importances in pth
        input:
                model_path: local path to model
                path_to_images: path to store the resulting image
                model_version: int; [optional] helps us retrive the last run_id of mlflow

        output:
                None
        """
        client = MlflowClient()

        # Defualt to previous run id, if current run_id is not given
        if self.mlflow_run_id is None:
            self.mlflow_run_id = client.get_model_version(
                "rf_model", model_version
            ).run_id
            self.logger.info(
                "Run_id is not defined. Default the run Id, to previous run_id (as indicated by last registered model version)."
            )
        check_if_directory_exits_and_create(pth=path_to_images, logger=self.logger)

        model_rf = mlflow.sklearn.load_model(f"{model_path}")
        self.logger.info("SUCCESS: Loaded model.")
        # Calculate feature importances
        importances = model_rf.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [self.X_train.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel("Importance")
        # Add bars
        plt.bar(range(self.X_train.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(self.X_train.shape[1]), names, rotation=90)
        plt.savefig(f"{path_to_images}/feature_importance.png")
        plt.close()
        self.logger.info("SUCCESS: Saved feature importnace plot to image path.")

        client.log_artifact(
            self.mlflow_run_id, f"{path_to_images}/feature_importance.png"
        )
        self.logger.info(
            "SUCCESS: Logged feature importance plot as artifact to mlflow run."
        )


if __name__ == "__main__":
    # Define parameters
    cat_columns = cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    response_names = [
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    # normalized_histogram, count_histogram, density_plot
    dict_columns_type_plots = {
        "Churn": ["count_histogram", "churn_distribution.png"],
        "Customer_Age": ["count_histogram", "customer_age_distribution.png"],
        "Marital_Status": ["normalized_histogram", "marital_status_distribution.png"],
        "Total_Trans_Ct": ["density_plot", "total_transaction_distribution.png"],
    }
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    param_grid_input = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    # Call Library functions

    churn_customers = ChurnLibrary(
        pth=r"./data/bank_data.csv", category_lst=cat_columns, logger=logger_cls
    )
    input_data_frame = churn_customers.import_data()
    input_data_frame_preprocessed = churn_customers.encoder_helper(
        response=response_names
    )

    churn_customers.perform_eda(
        dict_columns_to_plot_type=dict_columns_type_plots,
        path_to_storage="./images/eda",
        figure_size=(20, 10),
        include_heatmap=True,
    )
    churn_customers.perform_feature_engineering(response=keep_cols)
    churn_customers.train_models(
        param_grid=param_grid_input,
        path_to_images="./images/results",
    )
    churn_customers.feature_importance_plot(
        model_path="./rf_model", path_to_images="./images/results", model_version=3
    )
    churn_customers.classification_report_image(
        path_to_images="./images/results", model_version=3
    )
