from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import handling_outliers, handling_missing, feature_engineering, feature_dropping, feature_encoding


model = pickle.load(open('catboost_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = {
        'Year_Birth': [int(request.form['Year_Birth'])],
        'Education': [request.form['Education']],
        'Marital_Status': [request.form['Marital_Status']],
        'Income': [float(request.form['Income'])],
        'Kidhome': [int(request.form['Kidhome'])],
        'Teenhome': [int(request.form['Teenhome'])],
        'Dt_Customer': [request.form['Dt_Customer']],  # Assuming you handle this as a date later
        'Recency': [int(request.form['Recency'])],
        'MntWines': [int(request.form['MntWines'])],
        'MntFruits': [int(request.form['MntFruits'])],
        'MntMeatProducts': [int(request.form['MntMeatProducts'])],
        'MntFishProducts': [int(request.form['MntFishProducts'])],
        'MntSweetProducts': [int(request.form['MntSweetProducts'])],
        'MntGoldProds': [int(request.form['MntGoldProds'])],
        'NumDealsPurchases': [int(request.form['NumDealsPurchases'])],
        'NumWebPurchases': [int(request.form['NumWebPurchases'])],
        'NumCatalogPurchases': [int(request.form['NumCatalogPurchases'])],
        'NumStorePurchases': [int(request.form['NumStorePurchases'])],
        'NumWebVisitsMonth': [int(request.form['NumWebVisitsMonth'])],
        'Complain': [int(request.form['Complain'])]  # Assuming it's 0 or 1
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    with open('lgbm_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    X = feature_dropping(feature_engineering(feature_encoding(handling_outliers(handling_missing(df)))))
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:,1]
    
    
    return render_template('prediction.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)