# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:19:33 2021

@author: 14076
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import pandas_profiling
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

df_rough=pd.read_csv("Womens Clothing E-Commerce Reviews.csv",index_col=False)
column_names=['Clothing ID','Age','Title','Review Text','Rating','Recommended IND',
              'Positive Feedback Count','Division Name','Department Name','Class Name']
df=pd.DataFrame(data=df_rough,columns=column_names)
df.info()
round(df.describe())

# Delete missing observations for following variables
for x in ["Division Name","Department Name","Class Name","Review Text"]:
    df = df[df[x].notnull()]

# Create New Variables: 
# Word Length
df["Word Count"] = df['Review Text'].str.split().apply(len)
# Character Length
df["Character Count"] = df['Review Text'].apply(len)
# Classification for Positive and Negative Reviews
df['Sentiment Type'] = 'positive'
df.loc[df.Rating == 3,["Sentiment Type"]] = 'neutral'
df.loc[df.Rating < 3,["Sentiment Type"]] = 'negative'


# Extracting Missing Count and Unique Count by Column
unique_count = []
for x in df.columns:
    unique_count.append([x,len(df[x].unique()),df[x].isnull().sum()])
print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
pd.DataFrame(unique_count, columns=["Column","Unique","Missing"]).set_index("Column").T

df.describe().T.drop("count",axis=1)
df[["Title", "Division Name","Department Name","Class Name"]].describe(include=['O']).T.drop("count",axis=1)

# Continous Distributions
f, ax = plt.subplots(1,4,figsize=(16,4), sharey=False)
sns.distplot(df.Age, ax=ax[0], color='darkgreen')
ax[0].set_title("Age Distribution")
ax[0].set_ylabel("Density")
sns.distplot(df["Positive Feedback Count"], ax=ax[1], color='lightseagreen')
ax[1].set_title("Positive Feedback Count Distribution")
sns.distplot(df["Word Count"], ax=ax[2], color='lightseagreen')
ax[2].set_title("Word Count Distribution")
sns.distplot(df["Character Count"], ax=ax[3], color='lightseagreen')
ax[3].set_title("Character Count Distribution")
plt.tight_layout()
plt.show()

row_plots = ["Division Name","Department Name"]
f, axes = plt.subplots(1,len(row_plots), figsize=(14,4), sharex=False)
for i,x in enumerate(row_plots):
    sns.countplot(y=x, data=df,order=df[x].value_counts().index, ax=axes[i],
                  palette=sns.cubehelix_palette(8, start = 1, rot = -.75))
    axes[i].set_title("Count of Categories in {}".format(x))
    axes[i].set_xlabel("Frequency Count")
axes[0].set_ylabel("Category")
axes[1].set_ylabel("")
plt.show()

# Class Name
plt.subplots(figsize=(9,5))
sns.countplot(y="Class Name", data=df,order=df["Class Name"].value_counts().index,
              palette=sns.cubehelix_palette(len(df["Class Name"].unique()), start = 1, rot = -.75))
plt.title("Frequency Count of Class Name")
plt.xlabel("Count")
plt.show()

#cat_dtypes = [x for x,y,z in unique_count if y < 10 and x not in ["Division Name","Department Name"]]
cat_dtypes = ["Rating","Recommended IND","Sentiment Type"]
f, axes = plt.subplots(1,len(cat_dtypes), figsize=(14,4), sharex=False)
for i in range(len(cat_dtypes)):
    sns.countplot(x=cat_dtypes[i], data=df,order=df[cat_dtypes[i]].value_counts().index, 
                  ax=axes[i], palette=sns.cubehelix_palette(8, start = 1, rot = -.75))
    axes[i].set_title("Frequency Distribution")
    axes[i].set_ylabel("Occurrence")
    axes[i].set_xlabel("{}".format(cat_dtypes[i]))
axes[1].set_ylabel("")
axes[2].set_ylabel("")
plt.show()

# Plot Correlation Matrix
f, ax = plt.subplots(figsize=[9,6])
ax = sns.heatmap(df.corr(), annot=True,cmap=sns.diverging_palette(200,20,sep=10,as_cmap=True),
                 fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title("Correlation Matrix for All Variables")
plt.show()


import datetime as dt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import sklearn.metrics as mt
# from plotly import tools
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# import plotly.graph_objs as go


# fill NA values by space
df['Review Text'] = df['Review Text'].fillna('')

# CountVectorizer() converts a collection 
# of text documents to a matrix of token counts
vectorizer = CountVectorizer()

def wordcounts(s):
    c = {}
    # tokenize the string and continue, if it is not empty
    if vectorizer.build_analyzer(s):
        d = {}
        # find counts of the vocabularies and transform to array 
        w = vectorizer.fit_transform([s]).toarray()
        # vocabulary and index (index of w)
        vc = vectorizer.vocabulary_
        # items() transforms the dictionary's (word, index) tuple pairs
        for k,v in vc.items():
            d[v]=k # d -> index:word 
        for index,i in enumerate(w[0]):
            c[d[index]] = i # c -> word:count
    return  c

# add new column to the dataframe
df['Word Counts'] = df['Review Text'].apply(wordcounts)
df.head()

# selecting some words to examine detailed 
selectedwords = ['awesome','great','fantastic','extraordinary','amazing','super',
                 'magnificent','stunning','impressive','wonderful','breathtaking',
                 'love','content','pleased','happy','glad','satisfied','lucky',
                 'shocking','cheerful','wow','sad','unhappy','horrible','regret',
                 'bad','terrible','annoyed','disappointed','upset','awful','hate']

def selectedcount(dic,word):
    if word in dic:
        return dic[word]
    else:
        return 0

dfwc = df.copy()  
for word in selectedwords:
    dfwc[word] = dfwc['Word Counts'].apply(selectedcount,args=(word,))
    
word_sum = dfwc[selectedwords].sum()
print('Selected Words')
print(word_sum.sort_values(ascending=False).iloc[:5])

print('\nClass Names')
print(df['Class Name'].fillna("Empty").value_counts().iloc[:5])

fig, ax = plt.subplots(1,2,figsize=(20,10))
wc0 = WordCloud(background_color='white',
                      width=450,
                      height=400 ).generate_from_frequencies(word_sum)

cn = df['Class Name'].fillna(" ").value_counts()
wc1 = WordCloud(background_color='white',
                      width=450,
                      height=400 
                     ).generate_from_frequencies(cn)

ax[0].imshow(wc0)
ax[0].set_title('Selected Words\n',size=25)
ax[0].axis('off')

ax[1].imshow(wc1)
ax[1].set_title('Class Names\n',size=25)
ax[1].axis('off')

rt = df['Review Text']
plt.subplots(figsize=(18,6))
wordcloud = WordCloud(background_color='white',
                      width=900,
                      height=300
                     ).generate(" ".join(rt))
plt.imshow(wordcloud)
plt.title('All Words in the Reviews\n',size=25)
plt.axis('off')
plt.show()

t = df['Title'].fillna(" ")
plt.subplots(figsize=(18,6))
wordcloud = WordCloud(background_color='white',
                      width=900,
                      height=300
                     ).generate(" ".join(t))
plt.imshow(wordcloud)
plt.title('All Words in the Title\n',size=25)
plt.axis('off')
plt.show()


dfc = df[['Review Text','Rating']]
dfc = dfc[dfc['Rating'] != 3]
dfc['Sentiment'] = dfc['Rating'] >=4
dfc.head()

# split data
train_data,test_data = train_test_split(dfc,train_size=0.8,random_state=0)
# select the columns and 
# prepare data for the models 
X_train = vectorizer.fit_transform(train_data['Review Text'])
y_train = train_data['Sentiment']
X_test = vectorizer.transform(test_data['Review Text'])
y_test = test_data['Sentiment']

## fit the models
# logistic regression
start=dt.datetime.now()
lr = LogisticRegression()
lr.fit(X_train,y_train)
print('Elapsed time: ',str(dt.datetime.now()-start))

# multinomial naive bayes
start=dt.datetime.now()
nb = MultinomialNB()
nb.fit(X_train,y_train)
print('Elapsed time: ',str(dt.datetime.now()-start))


# define a dataframe for the predictions
dfp = train_data.copy()
dfp['Logistic Regression'] = lr.predict(X_train)
dfp['Naive Bayes'] = nb.predict(X_train)
dfp.head()

pred_lr = lr.predict_proba(X_test)[:,1]
fpr_lr,tpr_lr,_ = roc_curve(y_test,pred_lr)
roc_auc_lr = auc(fpr_lr,tpr_lr)

pred_nb = nb.predict_proba(X_test)[:,1]
fpr_nb,tpr_nb,_ = roc_curve(y_test.values,pred_nb)
roc_auc_nb = auc(fpr_nb,tpr_nb)

f, axes = plt.subplots(1, 2,figsize=(10,5))
axes[0].plot(fpr_lr, tpr_lr, color='darkred', lw=2, label='ROC curve (area = {:0.2f})'.format(roc_auc_lr))
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05])
axes[0].set(xlabel ='False Positive Rate', ylabel = 'True Positive Rate', title = 'Logistic Regression')
axes[0].legend(loc='lower right', fontsize=13)

axes[1].plot(fpr_nb, tpr_nb, color='darkred', lw=2, label='ROC curve (area = {:0.2f})'.format(roc_auc_nb))
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05])
axes[1].set(xlabel ='False Positive Rate', ylabel = 'True Positive Rate', title = 'Naive Bayes')
axes[1].legend(loc='lower right', fontsize=13)


# preparation for the confusion matrix
lr_cm=confusion_matrix(y_test.values, lr.predict(X_test),normalize='true')
nb_cm=confusion_matrix(y_test.values, nb.predict(X_test),normalize='true')

plt.figure(figsize=(15,12))
plt.suptitle("Confusion Matrices",fontsize=24)

plt.subplot(2,2,1)
plt.title("Logistic Regression")
sns.heatmap(lr_cm, annot = True, cmap=sns.color_palette('Blues'),cbar=False);

plt.subplot(2,2,2)
plt.title("Naive Bayes")
sns.heatmap(nb_cm, annot = True, cmap=sns.color_palette('Blues'),cbar=False);


print("Logistic Regression")
print(mt.classification_report(y_test, lr.predict(X_test)))
print("\n Naive Bayes")
print(mt.classification_report(y_test, nb.predict(X_test)))





