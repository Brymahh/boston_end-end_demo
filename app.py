import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
lr_ml = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    scaled_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    pred = lr_ml.predict(scaled_data)
    print(pred[0])

    return jsonify(pred[0])


@app.route('/predict', methods=['POST'])
def predict():
    data =[float(x) for x in request.form.values()]
    scaled_data = scalar.transform(np.array(data).reshape(1,-1))
    print(scaled_data)
    output =int(lr_ml.predict(scaled_data)[0])
    return render_template('home.html', prediction_text="The current Estimate of this house is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
