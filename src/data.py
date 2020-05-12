from typing import List, Dict
import pandas as pd


def get_train_test(features: List[str], imputation: Dict[str, str], min_deaths: int) -> Dict[str, pd.DataFrame]:
    """
    Returns a dictionary containing the train-test split of the features and labels

    Parameters
    features:  Names of features to retrieve. Possible options include just stationary for now.
    imputation: Imputation strategy for each retrieved feature. Possible options right now are 'median' for stationary.
    num_days: Number of days for each county to keep as the test set.

    """
