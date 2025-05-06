from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

def prepare_dataset():
    """
    Prepare dataset for training and testing.

    This function fetches the Iris dataset, splits it into train and test
    sets, and returns the four arrays.

    - The Iris dataset consists of 150 samples of iris flowers from three different species:
    Setosa, Versicolor, and Virginica.
    - The dataset has 4 features
      - Sepal Length: The length of the iris flower's sepals (the green leaf-like structures that encase the flower bud).
      - Sepal Width: The width of the iris flower's sepals.
      - Petal Length: The length of the iris flower's petals (the colored structures of the flower).
      - Petal Width: The width of the iris flower's petals.
    - X is the array of 150 samples each containing above 4 features
    - Y is the array of 0, 1, 2 which means whether species is Setosa, Versicolor and Virginica

    Parameters
    ----------
    None

    Returns
    -------
    x_train, x_test, y_train, y_test : tuple
        The Iris dataset split into training and test sets.
    """

    x, y = load_iris(return_X_y=True)

    # Split dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    return x_train, x_test, y_train, y_test


def train(x_train, y_train):
    """
    Train a model on the given dataset.

    This function creates a LogisticRegression model with certain parameters,
    fits it to the given dataset, and returns the model.

    Parameters
    ----------
    x_train, y_train : tuple
        The Iris dataset split into training and test sets.

    Returns
    -------
    logistic_regression_model : LogisticRegression
        The model which is trained with the given dataset
    """

    # Create model
    params = { "max_iter": 1000, "random_state": 8888, "multi_class": "auto", "solver": "newton-cg" }
    logistic_regression_model = LogisticRegression(**params)

    # Train model
    logistic_regression_model.fit(x_train, y_train)

    return logistic_regression_model


def predict(model, x_test):
    """
    Predict on the given test set.

    This function creates a prediction based on the given model and test set.

    Parameters
    ----------
    model : LogisticRegression
        The model which is trained with the given dataset
    x_test : tuple
        The Iris dataset split into training and test sets

    Returns
    -------
    y_predict : tuple
        The prediction based on the given model and test set
    """
    return model.predict(x_test)


def track_model(model, x_train, x_test, y_predict, accuracy, params):
    """
    Track model.

    This function tracks the given model in the MLflow server.

    Parameters
    ----------
    model : LogisticRegression
        The model which is trained with the given dataset
    x_train, x_test : tuple
        The Iris dataset split into training and test sets
    y_predict : tuple
        The prediction based on the given model and test set
    accuracy : float
        The accuracy of the model
    params : dict
        The parameters of the model
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Iris Experiment")

    with mlflow.start_run():
        # Set tag
        mlflow.set_tag("Training info", "Basic Logistic Regression model on Iris dataset")

        # log parameters
        mlflow.log_params(params)

        # log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Infer signature for model deployment.
        # This represents the input and output schema of the model
        model_signature = infer_signature(x_test, y_predict)

        # Logs a registered model as an MLflow artifact for the current run
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=model_signature,
            input_example=x_train,
            registered_model_name="iris-model"
        )

        print("Model Uri: {}".format(model_info.model_uri))

if __name__ == "__main__":
    # Prepare dataset
    x_train, x_test, y_train, y_test = prepare_dataset()

    # Train model
    logistic_regression_model = train(x_train, y_train)

    # Test model
    y_predict = predict(logistic_regression_model, x_test)

    # Track model
    accuracy = accuracy_score(y_test, y_predict)
    track_model(logistic_regression_model, x_train, x_test, y_predict, accuracy, logistic_regression_model.get_params(deep=True))
