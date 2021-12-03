from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, template_folder = 'templates')

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/load_data', methods = ['GET'], endpoint = 'load_data')
def load_data():

    test_data = pd.read_csv("test.csv", nrows = 5)

    n_columns  = test_data.iloc[: , 2:10]

    data_size  = test_data.shape

    return render_template('index.html', data_size = '{}'.format(data_size),
                           tables=[n_columns.to_html()], titles = ['Details'])

@app.route('/predict', methods = ['POST'], endpoint = 'predict')
def predict():

    test_data = pd.read_csv("test.csv", nrows = 5)
    n_columns  = test_data.iloc[: , 2:10]
    data_size  = test_data.shape

    model = open('flight_rf.pkl','rb')
    prediction_model = pickle.load(model)

    y_prediction = prediction_model.predict(test_data)

    result = [round(num, ndigits=2) for num in y_prediction]

    return render_template('index.html', tables=[n_columns.to_html()], titles = ['Details'],
                            result = '{}'.format(result), data_size = '{}'.format(data_size))

if __name__ == '__main__':
    app.run(debug = True, port=7001, host='localhost')