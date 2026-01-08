import flask
from flask import Flask,render_template, request
import numpy as np
import joblib

app = Flask(__name__)

RandomForest_model = joblib.load("RandomForest_pipeline.joblib")
Logistic_model = joblib.load("Logistic_pipeline.joblib")
KNeighbours_model =joblib.load("KNeighbours_pipeline.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/selection", methods =  ["GET"])


@app.route("/predict", methods = ["POST"])
def predict():
    sepal_length = float(request.form.get('sepal_length'))
    sepal_width = float(request.form.get('sepal_width'))
    petal_length = float(request.form.get('petal_length'))
    petal_width = float(request.form.get('petal_width'))

    model_name = request.form.get('action')

    if(model_name == "Random Forest"):
        prediction = RandomForest_model.predict([np.array([sepal_length, sepal_width, petal_length, petal_width])])[0]
        probabilities = RandomForest_model.predict_proba([np.array([sepal_length, sepal_width, petal_length, petal_width])])[0]
    elif(model_name == "Logistic Regression"):
        prediction = Logistic_model.predict([np.array([sepal_length, sepal_width, petal_length, petal_width])])[0]
        probabilities = Logistic_model.predict_proba([np.array([sepal_length, sepal_width, petal_length, petal_width])])[0]
    else:
        prediction = KNeighbours_model.predict([np.array([sepal_length, sepal_width, petal_length, petal_width])])[0]
        probabilities = KNeighbours_model.predict_proba([np.array([sepal_length, sepal_width, petal_length, petal_width])])[0]

    

   
    
    return render_template("result.html", prediction = prediction, setosa = np.round(probabilities[0], 4), 
                           versicolor = np.round(probabilities[1], 4), virginica = np.round(probabilities[2], 4))

@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")




if __name__ == '__main__':
    app.run(debug =True, host='0.0.0.0', port=5000)






