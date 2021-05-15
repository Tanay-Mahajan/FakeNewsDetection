from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np

lemmatizer = WordNetLemmatizer()
def preprocessing(news):
    lst = news.split()
    for j in range(len(lst)):
        if lst[j] == 'U.S.':
            lst[j] = "USA"

    check_news = " ".join(map(str, lst))

    final_check = re.sub('[^a-zA-Z]', ' ', check_news)
    final_check = final_check.lower()
    final_check = final_check.split()
    final_check = [lemmatizer.lemmatize(word) for word in final_check if not word in set(stopwords.words('english'))]
    final_check = ' '.join(final_check)
    voc_size = 10000
    final_onehot = one_hot(final_check, voc_size)

    final_onehot = np.array(final_onehot)
    final_onehot = final_onehot.reshape((1, len(final_onehot)))
    final_onehot = pad_sequences(final_onehot, padding='pre', maxlen=20)

    return final_onehot