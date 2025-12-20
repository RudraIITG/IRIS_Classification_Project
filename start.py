import flask
from flask import Flask,render_template, request
import numpy as np
import joblib

app = Flask(__name__)

iris_model = joblib.load("iris_pipeline.joblib")

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predict", methods = ["POST"])
def predict():
    sepal_length = float(request.form.get('sepal_length'))
    sepal_width = float(request.form.get('sepal_width'))
    petal_length = float(request.form.get('petal_length'))
    petal_width = float(request.form.get('petal_width'))

    prediction = iris_model.predict([np.array([sepal_length, sepal_width, petal_length, petal_width])])[0]

    probabilities = iris_model.predict_proba([np.array([sepal_length, sepal_width, petal_length, petal_width])])[0]
    
    return render_template("result.html", prediction = prediction, setosa = probabilities[0], versicolor = probabilities[1], virginica = probabilities[2])

@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")




if __name__ == '__main__':
    app.run(debug =True, host='0.0.0.0', port=5000)






