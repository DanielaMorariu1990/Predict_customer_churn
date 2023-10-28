import os
import logging
import pytest
from churn_library import *
from constants_pytest import (
    categorical_variables,
    response_names,
    dict_columns_type_plots,
    keep_cols,
    param_grid_input,
)


# Define the log file path and create the 'logs' directory if it doesn't exist
log_file_path = "./logs/churn_library_test.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Create a file handler and set the formatter
fileh = logging.FileHandler(log_file_path, "a")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fileh.setFormatter(formatter)

# Create a logger for the churn_library_test.log file
log = logging.getLogger("churn_library_test")
log.setLevel(logging.INFO)
log.addHandler(fileh)

log.info("TESTING ChurnLibrary:")


@pytest.fixture
def my_churn_class():
    return ChurnLibrary(
        pth=r"./data/bank_data.csv", category_lst=categorical_variables, logger=None
    )


@pytest.fixture
def set_up_churn_class(my_churn_class):
    my_churn_class.import_data()
    my_churn_class.encoder_helper(response=response_names)
    return my_churn_class


def test_import_data(my_churn_class):
    """
    test data import - this example is completed for you to assist with the other test functions
    """

    try:
        df = my_churn_class.import_data()
        log.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        log.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )

        raise err


def test_encoder_helper(set_up_churn_class):
    """
    test encoder helper
    """
    df = set_up_churn_class.data_frame.copy()
    number_cols = df.columns.isin(response_names).sum()
    try:
        assert number_cols == len(response_names)
        log.info("Response names are in resulting data frame!")
    except AssertionError as err:
        log.error(
            f"Response name are NOT in resulting data frame. We have {number_cols} encoded columns and need {len(response_names)}"
        )
        raise err


def test_perform_eda(set_up_churn_class):
    """
    test perform eda function
    """
    path = "./test_images/eda"
    set_up_churn_class.perform_eda(
        dict_columns_to_plot_type=dict_columns_type_plots,
        path_to_storage=path,
        figure_size=(20, 10),
        include_heatmap=True,
    )
    try:
        assert os.path.exists(path)
        log.info("SUCCESS: The test_impages/eda path was created!")

    except AssertionError as err:
        log.error("ERROR: The test_impages/eda path was not created!")

    try:
        files = os.listdir(path)
        assert len(files) == len(dict_columns_type_plots.keys()) + 1
        log.info("SUCCESS: All 5 plots have been created!")
    except AssertionError as err:
        log.error(
            f"ERROR: Number of files created :{len(files)} vs number of files that was requested: {len(dict_columns_type_plots.keys()) +1} dose NOT match. Test Failed!"
        )


def test_perform_feature_engineering(set_up_churn_class):
    """
    test perform_feature_engineering
    """
    set_up_churn_class.perform_feature_engineering(keep_cols)
    try:
        assert (
            set_up_churn_class.X_test.shape[0]
            == set_up_churn_class.data_frame.shape[0] * 0.3
        ) & (
            set_up_churn_class.X_test.shape[1]
            == set_up_churn_class.data_frame[keep_cols].shape[1]
        )
        log.info("SUCCESS: X_test shape is OK!")
    except AssertionError as err:
        log.error(
            f"ERROR: X_train shape is wrong. It should have {set_up_churn_class.data_frame.shape[0]*0.3} rows and {set_up_churn_class.data_frame[keep_cols].shape[1]} columns, instead it has {set_up_churn_class.X_test.shape[0]} rows and {set_up_churn_class.X_test.shape[1]} columns."
        )
    try:
        assert (
            set_up_churn_class.X_train.shape[0]
            == set_up_churn_class.data_frame.shape[0] * 0.7
        ) & (
            set_up_churn_class.X_train.shape[1]
            == set_up_churn_class.data_frame[keep_cols].shape[1]
        )
        log.info("SUCCESS: X_train shape is OK!")
    except AssertionError as err:
        log.error(
            f"ERROR: X_train shape is wrong. It should have {set_up_churn_class.data_frame.shape[0]*0.7} rows and {set_up_churn_class.data_frame[keep_cols].shape[1]} columns, instead it has {set_up_churn_class.X_test.shape[0]} rows and {set_up_churn_class.X_test.shape[1]} columns."
        )


def test_train_models(set_up_churn_class):
    """
    test train_models
    """
    set_up_churn_class.perform_feature_engineering(keep_cols)
    set_up_churn_class.train_models(
        param_grid=param_grid_input,
        path_to_images="./test_images/results",
        path_to_lr_model="lr_model_test",
        path_to_rf_model="rf_model_test",
    )
    try:
        files = os.listdir("./test_images/results")
        assert len(files) == 1
        log.info("SUCCESS: Saved one results file to desired path.")
    except AssertionError as err:
        log.error("ERROR: No results file was saved! ")
    try:
        assert os.path.exists("./lr_model_test/model.pkl")
        log.info("SUCCESS: LR model pickle file exists.")
    except AssertionError as err:
        log.error("ERROR: No LR model file created!")
    try:
        assert os.path.exists("./rf_model_test/model.pkl")
        log.info("SUCCESS: RF model pickle file exists.")
    except AssertionError as err:
        log.error("ERROR: No RF model file created!")
