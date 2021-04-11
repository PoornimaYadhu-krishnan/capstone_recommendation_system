



from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import xgboost

app = Flask(__name__)  # intitialize the flaks app  # common 


# loading the data 
 
preprocessed_df  = pd.read_csv("dataset/preprocessed.csv",index_col=0)


# index.html asking takes the user input

@app.route('/')
def Home():
    return render_template('index.html')

#Predict button on bthe UI is linked to this endpoint, teh business logic lies here.

#Load the pickle files and do process the data to be returned in view.html

@app.route('/product_reco/',methods = ['POST'])
def age_pred():
    user_input = request.form['fn']

    try:
        user_final_rating = pickle.load(open('pickle/recoEngine.pkl', 'rb'))
        tfidf_vect=pickle.load(open("pickle/tfidfVectorizer.pkl", "rb"))
        model=pickle.load(open("pickle/XGBClassifiermodel.pkl", "rb"))
        products = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
        df_products=products.to_frame()
        preprocessed_df.reset_index(drop=True,inplace = True)
        recommdation_df = preprocessed_df[preprocessed_df['name'].isin(df_products[user_input].index)]
        recommdation_df.reset_index(drop=True,inplace = True)
        filter_df=recommdation_df[['name','reviews_rating','reviews_username','user_sentiment','reviews']]
        filter_df.reset_index(drop=True,inplace = True)
        transform_df=tfidf_vect.transform(recommdation_df['reviews'])
        y_pred=model.predict(transform_df)
        concat_df= pd.DataFrame({'Predictions': y_pred})
        concat_df.reset_index(drop=True,inplace = True)
        filter_df['Predictions']=concat_df['Predictions']
        result_pred1 = filter_df.groupby('name').agg({'Predictions': 'mean'})
        result_final=result_pred1.sort_values(by='Predictions',ascending=False).iloc[:5,:] 
        result_list=list(result_final.index)
        return  render_template('view.html',predictions=result_list)
    
    except:
        return render_template('error.html')






# Any HTML template in Flask App render_template

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of your flask app or api
    
    #,host="0.0.0.0")






