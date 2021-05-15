from flask import Flask, render_template, request
#from flask_cors import CORS,cross_origin
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import preprocess as pp


app = Flask(__name__)
model1 = load_model("my_model.h5")


@app.route('/',methods=["GET"])
def homepage():
    return render_template("index.html")

@app.route('/predict',methods=["POST"])
def predict():
    check_news = str(request.form['check_news'])
    print(check_news)
    final = pp.preprocessing(check_news)
    # lst = check_news.split()
    # for j in range(len(lst)):
    #     if lst[j] == 'U.S.':
    #         lst[j] = "USA"
    #
    # check_news = " ".join(map(str, lst))
    #
    # final_check = re.sub('[^a-zA-Z]', ' ', check_news)
    # final_check = final_check.lower()
    # final_check = final_check.split()
    # final_check = [lemmatizer.lemmatize(word) for word in final_check if not word in set(stopwords.words('english'))]
    # final_check = ' '.join(final_check)
    # voc_size = 10000
    # final_onehot = one_hot(final_check, voc_size)
    #
    # final_onehot = np.array(final_onehot)
    # final_onehot = final_onehot.reshape((1, len(final_onehot)))
    # final_onehot = pad_sequences(final_onehot, padding='pre', maxlen=20)

    ans = model1.predict(final)
    if (np.round(ans) == 0):
        output = "News is true"
    else:
        output = "News is fake"


    return render_template("index.html",prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)