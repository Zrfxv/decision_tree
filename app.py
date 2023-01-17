# import Flask class and render_template
# An instance of this class will be our WSGI application.
from re import X
from flask import Flask
from flask.templating import render_template
from flask import request, url_for
import pickle
import numpy as np
import pandas as pd


# create an object of the class "Flask" by passing first argument.
app = Flask(__name__)
model = pickle.load(open("tree_model.pkl", "rb"))


# use the route() decorator to tell Flask what URL should trigger our function.
# argument here is pattern of url
@app.route("/")
def home():
    return render_template("index.html", output='belum diprediksi')


@app.route('/predict', methods=["POST"])
def predict():
    input_features = [float(X) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean',
                     'texture_se', 'area_se', 'texture_worst', 'smoothness_worst',
                     'compactness_worst', 'symmetry_worst']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 0:
        hasil = "Anda beresiko Kanker Jinak"
    else:
        hasil = "Anda beresiko Kanker Ganas"

    return render_template("index.html", output=hasil)


if __name__ == "__main__":
    app.run(debug=True)
