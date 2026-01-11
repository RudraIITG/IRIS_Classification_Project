from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

MODELS = {
    "Random Forest": joblib.load("RandomForest_pipeline.joblib"),
    "Logistic Regression": joblib.load("Logistic_pipeline.joblib"),
    "KNN": joblib.load("KNeighbours_pipeline.joblib"),
}

DEFAULT_MODEL_KEY = "KNN"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Parse once
    x = np.array([[
        float(request.form.get("sepal_length")),
        float(request.form.get("sepal_width")),
        float(request.form.get("petal_length")),
        float(request.form.get("petal_width")),
    ]], dtype=np.float32)

    model_key = request.form.get("action")
    model = MODELS.get(model_key, MODELS[DEFAULT_MODEL_KEY])

    prediction = model.predict(x)[0]
    probabilities = model.predict_proba(x)[0]

    return render_template(
        "result.html",
        prediction=prediction,
        setosa=np.round(probabilities[0], 4),
        versicolor=np.round(probabilities[1], 4),
        virginica=np.round(probabilities[2], 4),
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
