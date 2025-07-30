import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import joblib
import string

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake.head()

true.head()

fake['class']=0
true['class']=1

data = pd.concat([fake,true],axis=0)

data.sample(10)

data = data.drop(["title","subject","date"],axis=1)

data.reset_index(inplace=True)

data.drop(['index'],axis=1,inplace=True)

data.sample(5)

def clean_text(text):
    """Cleans and normalizes input text for further processing.
    
    This function removes unwanted characters, punctuation, numbers, and 
    formatting from the input text to prepare it for analysis.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned and normalized text.
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'\W', " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', "", text)
    text = re.sub(r'<.*?>+', "", text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub(r'\n', "", text)
    text = re.sub(r'\w*\d\w*', "", text)
    return text

data["text"] = data["text"].apply(clean_text)

x=data["text"]
y=data["class"]
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25,random_state=42)

vectorize = TfidfVectorizer()
xv_train = vectorize.fit_transform(xtrain)
xv_test = vectorize.transform(xtest)

lr = LogisticRegression()
lr.fit(xv_train,ytrain)

prediction = lr.predict(xv_test)
lr.score(xv_test,ytest)

print(classification_report(ytest,prediction))

joblib.dump(vectorize,'vectorize.joblib')
joblib.dump(lr,'model.joblib')