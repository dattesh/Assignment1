import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Loads the Boston Housing dataset manually since sklearn has deprecated it.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # Split into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # Target variable
    return df

def preprocess_data(df, test_size=0.2, random_state=42, scale=True):
    """
    Splits data into train/test and scales features if required.
    """
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train, X_test = X_train.values, X_test.values

    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Trains the model and evaluates it using Mean Squared Error (MSE).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse
