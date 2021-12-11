import numpy as np 
import pandas as pd
from flask import Flask,request,jsonify,render_template
import joblib
app=Flask(__name__)
model=joblib.load("Model.pkl")
tf_vector=joblib.load("tfvector.pkl")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = [request.form['review']]
    output=model.predict(tf_vector.transform(text))=='__label__2 '
    if output[0]==True:
        result='Positive Review'
    else:
        result='Negative Review'
    return render_template('analysis.html', prediction_text='Predicted Sentiment:   {}'.format(result))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict(tf_vector.transform(list(data.values())))
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)