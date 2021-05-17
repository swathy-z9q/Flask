# -*- coding: utf-8 -*-
"""
@author: Swathy Sekar
""" 
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import re 
import pickle



app = Flask(__name__)

#Loading pickles
user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
product_info = pickle.load(open('product_information.pkl','rb'))
customer_info = pickle.load(open('premium_customers.pkl','rb'))
trending = pickle.load(open('trending.pkl','rb'))
product_info = pickle.load(open('product_information.pkl', "rb"))
user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
bow_vectorizer = pickle.load(open('bow_vectorizer.pkl', "rb"))
dct = pickle.load(open('dct.pkl', "rb"))

product_info = product_info.sort_values('id', ascending=False)
product_info = product_info.drop_duplicates(subset='id', keep='first')

keys_list =list(product_info['id'])
values_list = list(product_info['name'])
name_zip_iterator = zip(keys_list, values_list)
get_name = dict(name_zip_iterator)

keys_list =list(product_info['id'])
values_list = list(product_info['reviews_text'])
review_zip_iterator = zip(keys_list, values_list)
get_reviews = dict(review_zip_iterator)

customers=list(customer_info['reviews_username'])


@app.route('/',methods =['GET','POST'])
def home():
    response = {}
    response['recommendations'] = []
    trending['score'] = np.round(trending['score'],2)
    response['trending'] = list(trending.to_records())
    response['user_name'] = ""
    return render_template('index.html', response=response,table=customers)

@app.route('/predict', methods =['POST'])
def predict():
    
    user_name = request.form.get("table")
    result = recommendations(user_name)

    trends = {}
    trends['products'] = list(trending['name'])
    trending['score'] = np.round(trending['score'],2)
    #Records name, score, average rating positive reviews
    response = {}
    response['recommendations'] = result
    response['trending'] = list(trending.to_records())
    response['user_name'] = user_name

    return render_template('index.html', response=response,table=customers)


def remove_pattern(text,pattern):
    
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text
        
def get_review_sentiment(text):    
    cleaned_text = remove_pattern(text, "@[\w]*")
    cleaned_text = cleaned_text.replace("[^a-zA-Z#]", " ")
    cleaned_text = ' '.join([w for w in cleaned_text.split() if len(w)>3])
    tokenized_review = cleaned_text.split()
    #tokenized_review = [ps.stem(i) for i in tokenized_review]
    tokenized_review = ' '.join(tokenized_review)
    bow_text = bow_vectorizer.transform([cleaned_text])
    sentiment = dct.predict(bow_text)[0]
    return sentiment

def recommendations(customer):

    d = pd.DataFrame(user_final_rating.loc[customer].sort_values(ascending=False)[0:20]).reset_index()
    d['products'] = d['id'].replace(get_name)
    d['reviews'] = d['id'].replace(get_reviews)
    reviews = list(d['reviews'])
    reviews_sentiment = []
    for review in reviews:
        reviews_sentiment.append(get_review_sentiment(review))
    d['reviews_sentiment'] = reviews_sentiment
    d = d.sort_values(by = 'reviews_sentiment',ascending = False)
    recommended_items = list(d['products'])[:5]
    return recommended_items


if __name__ == "__main__":
    app.run(debug=True)

