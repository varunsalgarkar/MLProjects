# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 20:23:45 2020

@author: vssal
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 19:51:34 2020

@author: vssal
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'diabPCKL.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('diabetesF.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        bmi = float(request.form['bmi'])
        insulin = int(request.form['insulin'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, bmi, insulin,  age]])
        my_prediction = classifier.predict(data)
        
        return render_template('resultDiab.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
