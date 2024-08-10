import pandas as pd
import numpy as np
import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dataframe = pd.read_csv('superstore_data.csv')

def handling_outliers(X:pd.DataFrame) -> pd.DataFrame:
    print(f"[{datetime.datetime.now()}] Handling Outliers Values processing started.")
    df = X.copy()
    for i in df.columns:
        if df[i].dtype != 'object' :
            data = dataframe[i].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            theoretical_upper = Q3 + 1.5 * IQR
            theoretical_lower = Q1 - 1.5 * IQR
            actual_upper = data[data <= theoretical_upper].max()
            actual_lower = data[data >= theoretical_lower].min()
            df[i] = np.where(df[i] > actual_upper, actual_upper, df[i])
            df[i] = np.where(df[i] < actual_lower, actual_lower, df[i])
    print(f"[{datetime.datetime.now()}] Handling Outliers Values processing completed successfully.")
    return df

def handling_missing(X:pd.DataFrame)->pd.DataFrame:
    print(f"[{datetime.datetime.now()}] Handling Missing Values processing started.")    
    df = X.copy()
    df['Income']=df['Income'].fillna(df['Income'].median())
    print(f"[{datetime.datetime.now()}] Handling Missing Values processing completed successfully.")
    return df

def feature_engineering(X:pd.DataFrame) -> pd.DataFrame:
    print(f"[{datetime.datetime.now()}] Feature Engineering processing started.")
    df = X.copy()
    df['Age'] = datetime.datetime.now().year - df['Year_Birth']
    df['AvgMnt'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']) / 6
    df['AvgNum'] = (df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']) / 4
    df['TotKids'] = df['Kidhome'] + df['Teenhome']
    df['Weeks_Lastbuy'] = df['Recency'] // 7
    
    dataframe["Income_Score"] = pd.qcut(dataframe['Income'].rank(method="first"), 5, labels=[1,2,3,4,5]).astype('category')
    df['Income_Score'] = dataframe['Income_Score']
    
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'],format="%Y-%m-%d")
    df['Dt_Customer'] =   pd.to_datetime(df['Dt_Customer'].dt.strftime('%m/%d/%Y'), format='%m/%d/%Y')
    df['Weeks_Enrollment'] = (datetime.datetime.now() - df['Dt_Customer']).dt.days // 7
    df['Income_PerCapita'] = df['Income'] / (df['TotKids'] + 1)
    print(f"[{datetime.datetime.now()}] Feature Engineering processing completed successfully.")
        
    return df

def feature_dropping(X:pd.DataFrame) -> pd.DataFrame:
    print(f"[{datetime.datetime.now()}] Feature Dropping processing started.")
    df = X.copy()
    df = df.drop([
                'Year_Birth',
                'Kidhome', 
                'Teenhome', 
                'Recency',
                'MntMeatProducts',
                'MntFishProducts',
                'MntSweetProducts',
                'MntFruits',
                'MntGoldProds',
                'MntWines',
                'NumDealsPurchases',
                'Dt_Customer',
                'Complain',
                    ],
                    axis=1)
    print(f"[{datetime.datetime.now()}] Feature Dropping processing completed successfully.")
    return df

education_map = {
    'PhD':4,
    'Master':3,
    'Graduation':2,
    '2n Cycle':1,
    'Basic':0
}
marital_map = {
    'Married':2,
    'Together':1,
    'Single':0,
    'Divorced':3,
    'Widow':4
}

def feature_encoding(X:pd.DataFrame) -> pd.DataFrame:
    print(f"[{datetime.datetime.now()}] Feature Encoding processing started.")
    df = X.copy()
    df.loc[df['Marital_Status'].isin(['YOLO', 'Alone', 'Absurd']), 'Marital_Status'] = 'Single'
    df['Education'] = df['Education'].map(education_map).astype('category')
    df['Marital_Status'] = df['Marital_Status'].map(marital_map).astype('category')
    df['Complain'] = df['Complain'].astype('category')
    print(f"[{datetime.datetime.now()}] Feature Encoding processing completed successfully.")
    return df