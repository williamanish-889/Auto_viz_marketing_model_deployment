import joblib
import warnings
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.base import InconsistentVersionWarning
import os

app = Flask(__name__)
# Suppress the specific InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load the model from the .pkl file


MODEL_PATH = os.path.join(os.getcwd(), "Auto_Viz_marketing_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Convert dictionary to DataFrame for model prediction
        # Assuming the model expects input in a specific structure,
        # for example, a DataFrame with specific column names.
        # Adjust column names based on your model's training data.
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)

        # Return prediction as JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # In a Colab environment, you'll need to run it with a public URL
    # For local development, you'd typically use app.run(debug=True)
    # For Colab, we'll use ngrok or a similar tool to expose the port.
    # However, for demonstration purposes, we'll just show the structure.
    print("To run this Flask app locally, save it as a .py file and execute 'python your_app_name.py'")
    print("Then, you can send POST requests to http://127.0.0.1:5000/predict")
    app.run(host="0.0.0.0", port=5000, debug=True)
