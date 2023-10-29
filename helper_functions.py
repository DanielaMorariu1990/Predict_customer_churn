"""
This is a helper function module for the class ChurnLibrary.
Author: Daniela Bielz
Date: 29.10.2023
"""
import os
import shutil
from sklearn.metrics import (
    plot_roc_curve,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


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
        if logger is not None:
            logger.info(f"Directory does not exist! Created directory: {pth}.")
    else:
        if logger is not None:
            logger.info(
                f"Directory {pth} already exists. Check completed, moving on..."
            )
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
                if logger is not None:
                    logger.info(
                        f"SUCCESS: Directory {model_pth} removed successfully. We need to delete local model directory for Mlflow process."
                    )

            except OSError as o:
                if logger is not None:
                    logger.exception(f"Error, {o.strerror}: {model_pth}")
                else:
                    raise o
        else:
            if logger is not None:
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
        if logger is not None:
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
