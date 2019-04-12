import pandas as pd
import numpy as np
import regex as re  ###Libraries Imported 


 import nltk
from nltk.corpus import stopwords ## Importing Stopword Library for removal of stopwords 
from nltk.stem import PorterStemmer ##Stemming
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  ## set language for stopwords of particular language 


data1 = pd.read_csv("2.txt",sep='\t',header=None) ##reading tab separated text file 
data1 = data1.drop_duplicates( keep='first', inplace=False) ##Dropping the duplicate records 
data1.columns = ["sentiment" , "review"] ##List made


data2 = pd.read_csv("1.txt", sep = "\t", header = None) 
data2 = data2.drop_duplicates( keep='first', inplace=False)
data2.columns = ["review", "sentiment"]
data2 = data2.reindex(columns = ["sentiment" , "review"]) ##Swapping the data frame coloumns 


data3 = pd.read_csv("3.txt", sep = ",") ##Reading tab separated test teext file 
data3 = pd.DataFrame(data3["Text"]) ##converting pandas series into Dataframe  
data3.columns = ['review']


len_train = len(data1)+len(data2)


data_fin = pd.concat([data1,data2])  ### final data to NLP is prepared
train_Y = data_fin["sentiment"]
train_Y = pd.DataFrame(train_Y)
del data_fin["sentiment"]


whole = pd.concat([data_fin,data3])
whole = whole.reset_index(drop = True)


corpus = []
for i in range(0,len(whole)):
    whole["review"][i] = re.sub("[^A-Za-z]", " ", str(whole["review"][i])).lower().split(" ")
    whole["review"][i] = [x for x in whole["review"][i] if x]
    whole["review"][i] = [ps.stem(w) for w in whole["review"][i] if not w in stop_words]
    whole["review"][i] = ' '.join(whole["review"][i])
    corpus.append(whole["review"][i])



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range = (1,2), max_features = 500) ###,max_df=0.50,min_df=0.004,
whole = tfidf.fit_transform(corpus).toarray()
train_new,test_new = whole[:len(data_fin),:],whole[len(data_fin):,:]



from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate = 0.1 , random_state = 0, n_estimators = 400, n_jobs = -1)
xgb.fit(train_new, train_Y)
pred_Y = xgb.predict(test_new)
pd.Dataframe(pred_Y).to_csv("final_prediction.csv")


### testing train and test 

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(train_new, train_Y, test_size=0.33, random_state=42)
xgb1 = XGBClassifier(learning_rate = 0.1 , random_state = 0, n_estimators = 400, n_jobs = -1)
xgb1.fit(X_train, y_train)
y_pred = xgb1.predict(X_test)
cm = confusion_matrix(y_test, y_pred)


#cross validation for k = 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb, train_new,train_Y, cv=10)
scores





















