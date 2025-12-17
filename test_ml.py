import pytest
# add necessary import
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference

# Pytest fixture to create a small sample dataset for testing
@pytest.fixture
def data():
    """
    Creates a simple dataframe for testing.
    """
    df = pd.DataFrame({
        'age': [30, 20, 50, 40],
        'workclass': ['Private', 'Private', 'Public', 'Private'],
        'education': ['Bachelors', 'Some-college', 'Masters', 'HS-grad'],
        'marital-status': ['Married', 'Single', 'Married', 'Divorced'],
        'occupation': ['Tech', 'Tech', 'Edu', 'Service'],
        'relationship': ['Husband', 'Own-child', 'Husband', 'Unmarried'],
        'race': ['White', 'Black', 'White', 'White'],
        'sex': ['Male', 'Female', 'Male', 'Female'],
        'native-country': ['USA', 'USA', 'USA', 'USA'],
        'salary': ['>50K', '<=50K', '>50K', '<=50K']
    })
    return df

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_shape(data):
    """
    Test that process_data returns the correct shapes.
    Verifies that the number of rows in X and y matches the input data.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    assert X.shape[0] == data.shape[0], "X should have same number of rows as input"
    assert y.shape[0] == data.shape[0], "y should have same number of rows as input"
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_process_data_targets(data):
    """
    Test that the processed targets are binary.
    Verifies that the label column is correctly encoded.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    
    _, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    assert np.all(np.isin(y, [0, 1])), "All targets must be 0 or 1"
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_inference_shape(data):
    """
    Test that inference returns predictions for every input row.
    Verifies the model produces a prediction for each sample.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    # Train a simple model on this dummy data
    model = train_model(X, y)
    
    # Run inference
    preds = inference(model, X)
    
    assert len(preds) == len(X), "Number of predictions should match number of inputs"
    pass
