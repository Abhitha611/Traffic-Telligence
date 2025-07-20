import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
model=pickle.load(open("model.pkl",'rb'))
scale = pickle.load(open('encoder.pkl','rb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("web_page.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result",methods=["POST"])
def result():
    input_feature=[float(x)for x in request.form.values()]
    features_values=[np.array(input_feature[0:11])]
    # features = features_values[:7]
    # print(features)
    names = [['holiday','temp','rain','snow','weather','year','month','day','hours','minutes','seconds']]
    data = pandas.DataFrame(features_values, columns=names)
    # data = scale.fit_transform(data)
    # data = pandas.DataFrame(data, columns = names)
    prediction=model.predict(data)
    print(prediction)
    text = "Estimated Traffic Volume is :"
    return render_template("result.html" ,prediction_text = text + str(prediction))


if __name__=="__main__":
    # port=8000, debug=True)
   
    app.run(debug=True,use_reloader=False)
# * running the app

# %%


# %%
import os
os.getcwd()

# %%
