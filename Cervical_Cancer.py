# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
# Importando Librerias y clases

import time

import pandas as pd # used to load the data
import numpy as np # optimized numerical library

from sklearn import preprocessing, metrics, utils, decomposition, model_selection, linear_model, discriminant_analysis, svm, tree, ensemble # library providing several ML algorithms and related utility
from imblearn import over_sampling # provides several resampling techniques to cope with unbalanced datasets (https://github.com/scikit-learn-contrib/imbalanced-learn) compatible with sklearn
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer, label_binarize

import matplotlib.pyplot as plt

# Libreria Boruta y paleta colores gradiente
from boruta import boruta_py
import seaborn as sns

# Start by defining three helper functions:
# - one to plot the sample distribution  acorss the class labels (to see how un-/balanced the dataset is)
# - one to compute and plot the confusion matrix
# - one to plot data in 2D with different colors per class label

def plot_pie(y, labels, title=""):
    target_stats = Counter(y)
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))

    fig, ax = plt.subplots()
    ax.set_title(title + " (size: %d)" % len(y))
    ax.pie(sizes, explode=explode, labels=target_stats.keys(), shadow=True, autopct='%1.1f%%')
    ax.axis('equal')

def cvClassifier(mdl, X, y, color, name, confMat = False, confMatNormalize = True):
    skf = StratifiedKFold(n_splits = 5)
    predicted_prob = np.zeros_like(y, dtype = float)
    for train,test in skf.split(X, y):
        mdl.fit(X[train,:],y[train])
        y_prob = mdl.predict_proba(X[test,:])
        predicted_prob[test] = y_prob[:,1] #The second class 1 from 0,1 is the one to be predicted
    
    precision, recall, thresholds = precision_recall_curve(y, predicted_prob)
    plt.plot(recall, precision, color=color,label = name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.legend()
    fscore = 2*(precision*recall)/(precision + recall)
    maxFidx = np.nanargmax(fscore)
    selP = precision[maxFidx]
    selRecall = recall[maxFidx]
    selThreshold = thresholds[maxFidx]

    return predicted_prob, selP, selRecall, fscore[maxFidx], selThreshold



def compute_and_plot_cm(ytest, ypred, labels, title=""):
    global nfigure
    # Compute confusion matrix
    cm = metrics.confusion_matrix(ytest, ypred)
    
    accuracy = metrics.accuracy_score(ytest, ypred, normalize=True)

    # Normalize the matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    # Plot the confusion matrix

    nfigure = nfigure + 1
    plt.figure(nfigure) # new numbered figure
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # plot the confusionmatrix using blue shaded colors
    plt.title("Confusion Matrix Normalized (%s) Accuracy: %.1f%%" % (title, accuracy*100)) # add title
    plt.colorbar() # plot the color bar as legend

    # Plot the x and y ticks using the class label names
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)


def plot_2d(xpred, ypred, labels, title=""):
    global nfigure
    # define the colors to use for each class label
    colors = ['red', 'blue', 'green', 'yellow', 'black']
    len_colors = len(colors)
    if len_colors < len(labels):
        print("WARNING: we have less colors than classes: some classes will reuse the same color")

    nfigure = nfigure + 1
    plt.figure(nfigure) # new numbered figure
    plt.title("Feature Space (%s)" % title) # add title


    # plot each class label with a separate color 
    for c in range(len(labels)):
        cur_class = (ypred == c) # get all points belonging to class c
        plt.plot(xpred[cur_class, 0], xpred[cur_class, 1], 'o', color=colors[c % len_colors]) # plot class c


nfigure = 0 #used to number the figures

# Aca viene el codigo 

# Cargar y Ver el data 
################ Load data ####################
# Get the dataset loaded and define class labels 
data = pd.read_csv(r'C:\Users\Recup\Documents\Big Data\cervical_cancer.csv', header=0)
data_class_labels = ["Level 1", "Level 2","Level 3","Level 4"]
data["Cancer Risk"] = data["Hinselmann"]+data["Schiller"]+data["Citology"]+data["Biopsy"]

# Replace ? with NAN
data = data.replace('?', np.nan)

data = data.convert_objects(convert_numeric=True) #turn data into numeric type for computation
#For continuous variable, we fill the median value. (THX for the suggestion in comment)
#For categorical variable, we fill 1.

# for continuous variable remplacing median value
data['Number of sexual partners'] = data['Number of sexual partners'].fillna(data['Number of sexual partners'].median())
data['First sexual intercourse'] = data['First sexual intercourse'].fillna(data['First sexual intercourse'].median())
data['Num of pregnancies'] = data['Num of pregnancies'].fillna(data['Num of pregnancies'].median())
data['Smokes'] = data['Smokes'].fillna(1)
data['Smokes (years)'] = data['Smokes (years)'].fillna(data['Smokes (years)'].median())
data['Smokes (packs/year)'] = data['Smokes (packs/year)'].fillna(data['Smokes (packs/year)'].median())
data['Hormonal Contraceptives'] = data['Hormonal Contraceptives'].fillna(1)
data['Hormonal Contraceptives (years)'] = data['Hormonal Contraceptives (years)'].fillna(data['Hormonal Contraceptives (years)'].median())
data['IUD'] = data['IUD'].fillna(0) # Under suggestion
data['IUD (years)'] = data['IUD (years)'].fillna(0) #Under suggestion
data['STDs'] = data['STDs'].fillna(1)
data['STDs (number)'] = data['STDs (number)'].fillna(data['STDs (number)'].median())
data['STDs:condylomatosis'] = data['STDs:condylomatosis'].fillna(data['STDs:condylomatosis'].median())
data['STDs:cervical condylomatosis'] = data['STDs:cervical condylomatosis'].fillna(data['STDs:cervical condylomatosis'].median())
data['STDs:vaginal condylomatosis'] = data['STDs:vaginal condylomatosis'].fillna(data['STDs:vaginal condylomatosis'].median())
data['STDs:vulvo-perineal condylomatosis'] = data['STDs:vulvo-perineal condylomatosis'].fillna(data['STDs:vulvo-perineal condylomatosis'].median())
data['STDs:syphilis'] = data['STDs:syphilis'].fillna(data['STDs:syphilis'].median())
data['STDs:pelvic inflammatory disease'] = data['STDs:pelvic inflammatory disease'].fillna(data['STDs:pelvic inflammatory disease'].median())
data['STDs:genital herpes'] = data['STDs:genital herpes'].fillna(data['STDs:genital herpes'].median())
data['STDs:molluscum contagiosum'] = data['STDs:molluscum contagiosum'].fillna(data['STDs:molluscum contagiosum'].median())
data['STDs:AIDS'] = data['STDs:AIDS'].fillna(data['STDs:AIDS'].median())
data['STDs:HIV'] = data['STDs:HIV'].fillna(data['STDs:HIV'].median())
data['STDs:Hepatitis B'] = data['STDs:Hepatitis B'].fillna(data['STDs:Hepatitis B'].median())
data['STDs:HPV'] = data['STDs:HPV'].fillna(data['STDs:HPV'].median())
data['STDs: Time since first diagnosis'] = data['STDs: Time since first diagnosis'].fillna(data['STDs: Time since first diagnosis'].median())
data['STDs: Time since last diagnosis'] = data['STDs: Time since last diagnosis'].fillna(data['STDs: Time since last diagnosis'].median())

# All data columns except last are input features (X), last column is output label (y)
n_features = len(data.columns) - 1

X = data.iloc[:,0:32].copy() 
Y = data.iloc[:,n_features].copy()

#

plot_pie(Y, data_class_labels, "Original") 
plt.show()

# Counting number of observations for Healthy and and Bankrupt Companies:
num_zeros = 0
num_ones = 0
num_two = 0
num_three = 0
num_four = 0
for num in Y:
       if num == 0:
              num_zeros = num_zeros+1
       if num == 1:
           num_ones = num_ones +1
       if num == 2:
           num_two = num_two +1 
       if num == 3:
        num_three = num_three + 1

num_four = len(Y) - num_zeros -num_ones -num_two -num_three

print("Numero de mujeres con Cancer Cervical nivel 1 :",num_ones)
print("Numero de mujeres con Cancer Cervical nivel 2 :",num_two)
print("Numero de mujeres con Cancer Cervical nivel 3 :",num_three)
print("Numero de mujeres con Cancer Cervical nivel 4 :",num_four)
print("Numero de mujeres sin Cancer Cervical:",num_zeros)

# Es posible notar que esta desbalanceado, por lo tanto tendremos que balancearlo

sm = over_sampling.SMOTE(random_state=42, ratio="auto")
X, Y = sm.fit_sample(X, Y)

# Plot the balanced label distribution
plot_pie(Y, data_class_labels, "Balanced")
plt.show()

################ Split data ####################
# Split data in training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.33, random_state=42)

################ Scale data ####################
# Train a scaler to standardize the features (zero mean and unit variance)
scaler = preprocessing.StandardScaler().fit(X_train)

# ... and scale the features QUE HACE ESTE ESCALADOR
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

################ PCA ####################
# Train a PCA with 2 dimensions
pca = decomposition.PCA(n_components=2).fit(X_train_scaled) 

# ... and apply it to the features
X_train_scaled_pca = pca.transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)

################ Logit ##################
# Train a Logit model on the original features
lr = linear_model.LogisticRegression().fit(X_train_scaled, y_train)

# Compute the predicted labels on test data
y_lr = lr.predict(X_test_scaled)
print("Acuracy of LR : %.1f%%" % (metrics.accuracy_score(y_test,y_lr)*100))

compute_and_plot_cm(y_test, y_lr, data_class_labels, title="LR")
plt.show()

## Es posible notar que la regresion logistica tiene un buen apunte para este tipo de problemas 

# Train a Logit model on pca extracted features
lr_pca = linear_model.LogisticRegression().fit(X_train_scaled_pca, y_train)

# Compute the predicted labels on test data
y_lr_pca = lr_pca.predict(X_test_scaled_pca)

print("Acuracy of LR + PCA: %.1f%%" % (metrics.accuracy_score(y_test,y_lr_pca)*100))

compute_and_plot_cm(y_test, y_lr_pca, data_class_labels, title="LR + PCA")

plot_2d(X_test_scaled_pca, y_lr_pca, data_class_labels, title="LR + PCA") # Linear Regression and Principal Component Analysis
plt.show()

################ QDA ##################
# Train a QDA model on original features
qda = discriminant_analysis.QuadraticDiscriminantAnalysis().fit(X_train_scaled, y_train)

# Compute the predicted labels on test data
y_qda = qda.predict(X_test_scaled)

print("Acuracy of QDA : %.1f%%" % (metrics.accuracy_score(y_test,y_qda)*100))

compute_and_plot_cm(y_test, y_qda, data_class_labels, title="LDA")
plt.show()

##Train a QDA model on pca extracted features
qda_pca = discriminant_analysis.QuadraticDiscriminantAnalysis().fit(X_train_scaled_pca, y_train)

# Compute the predicted labels on test data
y_qda_pca = qda_pca.predict(X_test_scaled_pca)

print("Acuracy of QDA + PCA: %.1f%%" % (metrics.accuracy_score(y_test,y_qda_pca)*100))

compute_and_plot_cm(y_test, y_qda_pca, data_class_labels, title="QDA + PCA")

plot_2d(X_test_scaled_pca, y_qda_pca, data_class_labels, title="QDA + PCA")
plt.show()

################ Polynomial expanded features ##################
# Train a polynomial expansion on original features
poly2 = preprocessing.PolynomialFeatures(degree=2).fit(X_train_scaled_pca)

# ... and apply it to the features
X_train_scaled_poly2 = poly2.transform(X_train_scaled_pca)
X_test_scaled_poly2 = poly2.transform(X_test_scaled_pca)

################  LDA on expanded ##################
# Train an LDA model on the original expanded features
lda_poly2 = discriminant_analysis.LinearDiscriminantAnalysis().fit(X_train_scaled_poly2, y_train)

# Compute the predicted labels on test data
y_lda_poly2= lda_poly2.predict(X_test_scaled_poly2)

print("Acuracy of ELDA: %.1f%%" % (metrics.accuracy_score(y_test,y_lda_poly2)*100))

compute_and_plot_cm(y_test, y_lda_poly2, data_class_labels, title="ELDA")

plot_2d(X_test_scaled_pca, y_lda_poly2, data_class_labels, title="ELDA")
plt.show()
################ SVM ##################
# Train a SVM model on the original features
sv = svm.SVC().fit(X_train_scaled, y_train)

# Compute the predicted labels on test data
y_sv = sv.predict(X_test_scaled)
print("Acuracy of SVM : %.1f%%" % (metrics.accuracy_score(y_test,y_sv)*100))

# Show confusion matrix
compute_and_plot_cm(y_test, y_sv, data_class_labels, title="SVM")

# Train a SVM model on PCA extracted features
sv_pca = svm.SVC().fit(X_train_scaled_pca, y_train)

# Compute the predicted labels on test data
y_sv_pca = sv_pca.predict(X_test_scaled_pca)
print("Acuracy of SVM +PCA: %.1f%%" % (metrics.accuracy_score(y_test,y_sv_pca)*100))

# Show confusion matrix
compute_and_plot_cm(y_test, y_sv_pca, data_class_labels, title="SVM + PCA")

# Show data in 2D
plot_2d(X_test_scaled_pca, y_sv_pca, data_class_labels, title="SVM + PCA")
plt.show()

################ DecisionTree ##################
# Train a DT model on the original features
dt = tree.DecisionTreeClassifier(max_depth=5).fit(X_train_scaled, y_train)

# Compute the predicted labels on test data
y_dt = dt.predict(X_test_scaled)
print("Acuracy of DecisionTree: %.1f%%" % (metrics.accuracy_score(y_test,y_dt)*100))

# Show confusion matrix
compute_and_plot_cm(y_test, y_dt, data_class_labels, title="DT")

# Train a DT model on PCA extracted features
dt_pca = tree.DecisionTreeClassifier(max_depth=5).fit(X_train_scaled_pca, y_train)


# Compute the predicted labels on test data
y_dt_pca = dt_pca.predict(X_test_scaled_pca)

print("Acuracy of DecisionTree +PCA: %.1f%%" % (metrics.accuracy_score(y_test,y_dt_pca)*100))

# Show confusion matrix
compute_and_plot_cm(y_test, y_dt_pca, data_class_labels, title="DT + PCA")

# Show data in 2D
plot_2d(X_test_scaled_pca, y_dt_pca, data_class_labels, title="DT + PCA")
plt.show()

## Feature the Importance Variables

forest = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)

forest.fit(X,Y)

for name, importance in zip(data, forest.feature_importances_):
    print(name, "=", importance)

features = data
importances = forest.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices],color = [(x/50.0, x/100.0, 0.75) for x in range(len(indices))], align='center')
plt.yticks(range(len(indices)),features.columns[indices])
plt.xlabel('Relative Importance')
plt.show()

###
# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
# Run classifier
Y2 = label_binarize(Y,classes=(0,1,2,3,4))
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y2, test_size=0.33, random_state=42)

################ Scale data ####################
# Train a scaler to standardize the features (zero mean and unit variance)
scaler = preprocessing.StandardScaler().fit(X_train)

# ... and scale the features QUE HACE ESTE ESCALADOR
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
random_state = np.random.RandomState(0)
n_classes = 5
fig, ax = plt.subplots()

def precision_recall(classifier,X_train_scaled,y_train,name,color):
    classifier.fit(X_train_scaled, y_train)
    y_score = classifier.decision_function(X_test_scaled)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], thresholds = precision_recall_curve(Y_test.ravel(),y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,average="micro") ## Aca se halla la prediccion promedio de las clases del modelo
    #plt.figure()
    ax.step(recall["micro"], precision["micro"], color=color,label = name ) #:{0:0.2f}.format(average_precision["micro"])
    ax.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,color ="none")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall Average')
    plt.legend()

# Aca se van graficando los precision recall promedio 
classifier = OneVsRestClassifier(linear_model.LogisticRegression(random_state=random_state))
precision_recall(classifier,X_train_scaled,y_train,"Logit","y")

classifier = OneVsRestClassifier(discriminant_analysis.QuadraticDiscriminantAnalysis())
precision_recall(classifier,X_train_scaled,y_train,"QDA","b")

classifier =OneVsRestClassifier(discriminant_analysis.LinearDiscriminantAnalysis())
precision_recall(classifier,X_train_scaled,y_train,"LDA","r")

classifier = OneVsRestClassifier(svm.SVC())
precision_recall(classifier,X_train_scaled,y_train,"SVC","g")

# Vamos a probar para decision tree

classifier =tree.DecisionTreeClassifier(max_depth=5)
classifier.fit(X_train_scaled, y_train)
y_score = classifier.predict_proba(X_test_scaled)
    
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], thresholds = precision_recall_curve(Y_test.ravel(),y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,average="micro") ## Aca se halla la prediccion promedio de las clases del modelo

ax.step(recall["micro"], precision["micro"], color="purple",label = "Decision Tree" ) #:{0:0.2f}.format(average_precision["micro"])
ax.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,color ="none")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision Recall Average')
plt.legend()


plt.show()

