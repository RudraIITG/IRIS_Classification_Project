from flask import Flask, render_template, request
import numpy as np
import joblib
from functools import lru_cache
import logging

app = Flask(__name__)

MODELS = {
    "Random Forest":"RandomForest_pipeline.joblib",
    "Logistic Regression":"Logistic_pipeline.joblib",
    "K Nearest Neighbours": "KNeighbours_pipeline.joblib"
}

logger = logging.getLogger("ML Models")
logger.setLevel(level=logging.INFO)

fileHandler  =logging.FileHandler("Model_logs.txt")
fileHandler.setLevel(logging.INFO)

formatter = logging.Formatter(fmt= "%(asctime)s | %(levelname)s | %(message)s")
fileHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.propagate = False



@lru_cache(maxsize=None)
def get_model(model_name):
    logger.info(msg="{} model is getting loaded".format(model_name))
    return joblib.load(MODELS.get(model_name))


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
    model  = get_model(model_key)
    if(model is None):
        logger.warning(msg="{} model could not load".format(model_key))
    else:
        if(get_model.cache_info().hits  == 0):
            logger.info(msg="{} model is loaded from disk".format(model_key))
        else:
            logger.info("{} model is loaded from the Cache".format(model_key))

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
