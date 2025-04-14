from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

filename = 'BREAST_CANCER.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
 'mean smoothness','mean compactness','mean concavity',
 'mean concave points', 'mean symmetry' ,'mean fractal dimension',
 'radius error','texture error' ,'perimeter error' ,'area error',
 'smoothness error','compactness error' ,'concavity error',
 'concave points error','symmetry error', 'fractal dimension error',
 'worst radius','worst texture' ,'worst perimeter' ,'worst area',
 'worst smoothness','worst compactness', 'worst concavity',
 'worst concave points','worst symmetry' ,'worst fractal dimension']
    df = pd.DataFrame(features_value, columns=features_name)
    output = classifier.predict(df)
    return render_template('result.html', prediction=output)

if __name__ == '__main__':
	app.run(debug=True)