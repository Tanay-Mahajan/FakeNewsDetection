import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk


from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential , load_model
import preprocess as pp

#reading training and test dataset
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

#adding label to dataset
fake_data['label']=1
true_data['label']=0

#merging both dataset row wise
data = pd.concat([true_data,fake_data],axis=0,ignore_index=True)
#created copy of dataset
x=data.copy()
y= x['label']
#removing the column used in prediction
x= x.drop('label',axis=1)

x['content'] = x['title'] + x['text']
x['content']

# preprocessing and cleaning data
# ps = PorterStemmer()

corpus = []

for i in range(len(x['title'])):
    # final =[]
    #final.append(op)
    op = pp.preprocessing(x['title'][i])
    corpus.append(op)
    # lst = x['title'][i].split()
    # for j in range(len(lst)):
    #     if lst[j] == 'U.S.':
    #         lst[j] = "USA"
    #
    # x['title'][i] = " ".join(map(str, lst))
    #
    # review = re.sub('[^a-zA-Z]', ' ', x['title'][i])
    # review = review.lower()
    # review = review.split()
    # review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    # review = ' '.join(review)
    #corpus.append(review)
    print(i, " row completed.")

corpus = np.array(corpus)
corpus = corpus.reshape(len(x['title']),20)
#converting text data into neumeric
voc_size=10000
# onehot_rep = [one_hot(word,voc_size)  for word in corpus]
#
#
sentlen =20
# embedding_doc = pad_sequences(onehot_rep,padding='pre',maxlen=sentlen)


#Actual model implementing
embedding_feature = 60
model = Sequential()
model.add(Embedding(voc_size,embedding_feature,input_length=sentlen))
model.add(LSTM(64,return_sequences=True))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(LSTM(32))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(Dense(1,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

x_final = np.array(corpus)

#spliting data
from sklearn.model_selection import train_test_split
x_temp, x_test, y_temp, y_test = train_test_split(x_final,y,test_size=0.2,random_state=1)
x_train,x_val,y_train,y_val = train_test_split(x_temp,y_temp,test_size=0.25,random_state=101)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
#Early stopiing means if loss value is not decreasing from min_delta then stop training

#actual training
history = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=5,batch_size=128,callbacks=[early_stopping])


#saving model
model.save('my_model.h5')