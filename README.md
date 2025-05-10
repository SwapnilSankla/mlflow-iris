## MLFlow playground

- Project is generated using: `poetry new mlflow-playground/`
- Start MLFlow Tracking server: `poetry run mlflow ui`
- Train model: Run `poetry run python src/mlflow_iris/train_iris_model.py` from command line from `mlflow-playground` directory
- Invoke inference: Run `poetry run python src/mlflow_playground/inference_iris_model.py` from command line from `mlflow-playground` directory
- Code is written on top of Iris dataset.
- Using GridSearchCV to perform hyperparameter tuning and choose the best model 


### Iris dataset

- The Iris dataset consists of 150 samples of iris flowers from three different species:
    Setosa, Versicolor, and Virginica.
- The dataset has 4 features
  - Sepal Length: The length of the iris flower's sepals (the green leaf-like structures that encase the flower bud).
  - Sepal Width: The width of the iris flower's sepals.
  - Petal Length: The length of the iris flower's petals (the colored structures of the flower).
  - Petal Width: The width of the iris flower's petals.
- X is the array of 150 samples each containing above 4 features
- Y is the array of 0, 1, 2 which means whether species is Setosa, Versicolor and Virginica