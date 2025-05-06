import mlflow

if __name__ == "__main__":
    model_name = "iris-model"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"

    mlflow.set_tracking_uri("http://localhost:5000")
    loaded_model = mlflow.sklearn.load_model(model_uri)

    print(loaded_model.predict([[2.1, 1.5, 10.4, 0.2]]))
