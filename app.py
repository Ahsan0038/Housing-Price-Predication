from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(
    open("california_housing_model.pkl", "rb")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array([[
        float(data["income"]),
        float(data["age"]),
        float(data["rooms"]),
        float(data["population"]),
        float(data["latitude"]),
        float(data["longitude"])
    ]])

    prediction = model.predict(features)[0]

    # Convert to dollars
    prediction = prediction * 100000

    return jsonify({
        "prediction": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)