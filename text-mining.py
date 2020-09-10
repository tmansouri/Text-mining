from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_train.target_names
['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
#print(len(twenty_train.data))
#print(len(twenty_train.filenames))
#print("\n".join(twenty_train.data[0].split("\n")[:3]))
#print(twenty_train.target_names[twenty_train.target[0]])
#print(twenty_train.target[:10])
#for t in twenty_train.target[:10]:
# print(twenty_train.target_names[t])
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
print(X_train_counts.shape)
count_vect.vocabulary_.get(u'algorithm')
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
#print(X_train_tf)
tfidf_transformer = TfidfTransformer()
data = tfidf_transformer.fit_transform(X_train_counts)
print(data.shape)
#data.corr(method ='pearson')
# imputation sur la base des 3 plus proches voisins sans données manquantes
#  obtenus à partir de distances calculées avec les seules données présentes
   # vérification
# On mélange les données
#///////////////////////////
# Suppression de certaines valeurs de la seconde variable (colonne)
#  pour obtenir des données manquantes
n_samples = data.shape[0]
# définition du taux de lignes à valeurs manquantes
missing_rate = 0.3
n_missing_samples = int(np.floor(n_samples * missing_rate))
print("Nous allons supprimer {} valeurs".format(n_missing_samples))
# choix des lignes à valeurs manquantes
present = np.zeros(n_samples - n_missing_samples, dtype=np.bool)
missing = np.ones(n_missing_samples, dtype=np.bool)
missing_samples = np.concatenate((present, missing))
# On mélange le tableau des valeurs absentes
np.random.shuffle(missing_samples)
print(missing_samples)
print(missing_samples.shape)
# obtenir la matrice avec données manquantes : manque indiqué par
#  valeurs NaN  dans la seconde colonne pour les lignes True dans
#   missing_samples
data_missing = data.copy()
data_missing[np.where(missing_samples), 1] = np.nan
print(data_missing.shape)
#print(stats.describe(data_missing))
# imputation par la moyenne: les NaN sont remplacés par la moyenne
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data_imputed = imp.fit_transform(data_missing)
#print("la")
#df = pd.DataFrame(columns=['l','c','v'])
#for i, line in enumerate(data):
#    for j, col in enumerate(line):
#    	print(i, j)
#    	new_row = {'l':i,'c':j,'v':col}
#    	df.append(new_row, ignore_index=True)
df = pd.DataFrame(data)
print(df)
print(df.shape)
clf = MultinomialNB().fit(data_imputed, twenty_train.target)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
 print('%r => %s' % (doc, twenty_train.target_names[category]))
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
])
text_clf.fit(twenty_train.data, twenty_train.target)
print(twenty_train.target)
print(twenty_train.target.shape)
twenty_test = fetch_20newsgroups(subset='test',
     categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),
 ])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
from sklearn import tree
clf = tree.DecisionTreeClassifier()
X = data
y = twenty_train.target
clf = clf.fit(X, y)
print(np.bincount(y))
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
clf.fit(X_train, y_train)
tree.plot_tree(clf, filled=True)
with open("arbre.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f, filled=True)
import pydot
(graph,) = pydot.graph_from_dot_file('arbre.dot')
graph.write_png('somefile.png')
Vous avez répondu à Sadĕk
from sklearn.datasets import fetch_20newsgroups from sklearn.feature_ext…
