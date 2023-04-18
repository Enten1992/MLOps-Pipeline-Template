
import mlflow
import os
from flask import Flask, request, jsonify
import json

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "MyMLModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class ModelServer:
    def __init__(self):
        print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
        try:
            self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        self.app = Flask(__name__)
        self.app.route("/predict", methods=["POST"])(self.predict)
        self.app.route("/health", methods=["GET"])(self.health_check)

    def predict(self):
        if self.model is None:
            return jsonify({"error": "Model not loaded"}), 500

        try:
            data = request.get_json(force=True)
            predictions = self.model.predict(data)
            return jsonify(predictions.tolist() if hasattr(predictions, 'tolist') else predictions)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    def health_check(self):
        return jsonify({"status": "healthy", "model_loaded": self.model is not None})

    def run(self, host="0.0.0.0", port=5001):
        print(f"Starting Flask server on {host}:{port}")
        self.app.run(host=host, port=port)

if __name__ == "__main__":
    # Example of how to run the server
    # For demonstration, ensure an MLflow tracking server is running and a model is registered.
    # Example: 
    # mlflow server --host 0.0.0.0 --port 5000
    # Then register a dummy model:
    # with mlflow.start_run():
    #     mlflow.log_param("param1", 1)
    #     mlflow.pyfunc.log_model("model", python_model=lambda x: x, registered_model_name="MyMLModel")
    # mlflow.register_model(model_uri="runs:/<run_id>/model", name="MyMLModel")
    # mlflow.transition_model_version_stage(name="MyMLModel", version=1, stage="Production")

    # To run this script:
    # python src/model_deployment.py
    # or with environment variables:
    # MLFLOW_TRACKING_URI=http://localhost:5000 MODEL_NAME=MyMLModel MODEL_STAGE=Production python src/model_deployment.py
    
    server = ModelServer()
    server.run()
