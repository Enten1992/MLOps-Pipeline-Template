import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_preprocess_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format=\"%(asctime)s - %(levelname)s - %(message)s\")

def train_model(n_estimators=100, max_depth=10, random_state=42):
    """
    Trains a RandomForestClassifier model, logs parameters and metrics to MLflow.

    Args:
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the tree.
        random_state (int): Controls the randomness of the estimator.
    """
    logging.info("Starting model training process...")

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    logging.info("Data loaded successfully.")

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        logging.info(f"Logged parameters: n_estimators={n_estimators}, max_depth={max_depth}")

        # Initialize and train the model
        logging.info("Initializing and training RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        logging.info("Model training complete.")

        # Make predictions
        logging.info("Making predictions on the test set...")
        y_pred = model.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=\"weighted\")
        recall = recall_score(y_test, y_pred, average=\"weighted\")
        f1 = f1_score(y_test, y_pred, average=\"weighted\")
        logging.info(f"Model evaluation complete. Accuracy: {accuracy:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        logging.info("Logged metrics to MLflow.")

        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        logging.info("Model logged to MLflow.")

        logging.info("MLflow Run completed successfully.")

if __name__ == \"__main__\":
    # Example of how to run the training with custom parameters
    # You can also pass these as command-line arguments if desired
    train_model(n_estimators=150, max_depth=12)
    print("\nTo view MLflow UI, run `mlflow ui` in your terminal and navigate to http://localhost:5000")
