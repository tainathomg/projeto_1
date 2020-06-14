# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


df_train = pd.read_csv("Ref_train.csv",  delimiter=';')
df_test = pd.read_csv("Ref_test.csv",  delimiter=';')

X_train = df_train.drop(['Classe', 'Amostra', 'Ponto'], axis='columns')
Y_train = df_train.Classe

X_test = df_test.drop(['Classe', 'Amostra', 'Ponto'], axis='columns')
Y_test = df_test.Classe



def svm():
    svm = SVC(gamma='scale', kernel='poly', C=20)
    svm.fit(X_train, Y_train)
    Y_pred_svm = svm.predict(X_test)
    matriz_svm = pd.crosstab(Y_test,Y_pred_svm, rownames=['Real'], colnames=['Predito'], margins=True)
    metrics_svm = metrics.classification_report(Y_test,Y_pred_svm)
    return print('Método SVM'), print('MATRIZ DE CONFUSÃO'), print(matriz_svm), print(' '), print('MÉTRICAS'), print(metrics_svm), print('******'), print()


def knn():
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train,Y_train)
    Y_pred_knn = knn.predict(X_test)
    matriz_knn = pd.crosstab(Y_test,Y_pred_knn, rownames=['Real'], colnames=['Predito'], margins=True)
    metrics_knn = metrics.classification_report(Y_test,Y_pred_knn)
    return print('MÉTODO KNN'), print(' '), print('Matriz de confusão'), print(matriz_knn), print(' '), print('Métricas'), print(metrics_knn), print('******'), print()

def rf():
    rf = RandomForestClassifier(n_estimators=13)
    rf.fit(X_train,Y_train) 
    Y_pred_rf = rf.predict(X_test)
    matriz_rf = pd.crosstab(Y_test,Y_pred_rf, rownames=['Real'], colnames=['Predito'], margins=True)
    metrics_rf = metrics.classification_report(Y_test,Y_pred_rf)
    return print('MÉTODO RANDOM FOREST'), print(' '), print('Matriz de confusão'), print(matriz_rf), print(' '), print('Métricas'), print(metrics_rf), print('******'), print()


svm()
knn()
rf()



    