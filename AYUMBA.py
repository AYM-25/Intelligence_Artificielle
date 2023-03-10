import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.preprocessing  import MinMaxScaler
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC

import os

with st.sidebar:
    st.title("IA Learning")
    choice = st.radio(
        "Navigation", ["Regression", "SVM", "Bayes", "DecisionTree", "Cluster", "PCA_Malll"])
    st.info("Faites un choix")

if os.path.exists("Sourcedata.csv"):
    df = pd.read_csv("Sourcedata.csv", index_col=None)

if choice == "Regression":
    st.title("Telecharger votre fichier de donnees ici")
    file = st.file_uploader("Telechargement de donnees")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("SourcedataR.csv", index=None)
        st.dataframe(df)

    numeric_cols = df.select_dtypes(exclude="object").columns.to_list()
    var_x = st.selectbox("choisis la variable en Abscisse", numeric_cols)
    var_y = st.selectbox("choisis la variable en Ordonnee", numeric_cols)

    fig = px.scatter(
        data_frame=df,
        x=var_x,
        y=var_y,
        title=str(var_y) + " vs " + str(var_x)
    )
    st.plotly_chart(fig)
    st.line_chart(df)

    X = df.iloc[:, 0].values
    Y = df.iloc[:, -1].values

    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(-1, 1)
    Y = np.array(Y)
    Y = Y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    linearRegr = LinearRegression()
    linearRegr.fit(X_train, y_train)
    pred_train = linearRegr.predict(X_test)

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    plt.plot(X_train, linearRegr.predict(X_train), color='blue')
    plt.scatter(df.iloc[:, 0].values,
                df.iloc[:, -1].values,
                color="red")
    st.header("Regression Training set")
    st.write(fig2)

    fig3 = plt.figure()

    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, linearRegr.predict(X_train), color='blue')

    st.header("Regression Test set")
    st.write(fig3)

if choice == "SVM":

    cancer = datasets.load_breast_cancer()

    st.write("Features :", cancer.feature_names)

    st.write("Labels :", cancer.target_names)

    st.write(cancer.data.shape)

    st.write(cancer.data[0:5])

    st.write(cancer.target)

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.3, random_state=109)

    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    st.write("Precision:", metrics.precision_score(y_test, y_pred))

    st.write("Recall:", metrics.recall_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    st.write(cm)
    pass

if choice == "Bayes":
	st.title("Telecharger votre fichier de donnees ici")
	file = st.file_uploader("Telechargement de donnees")
	if file:
			df1 = pd.read_csv(file, index_col=None)
			df1.to_csv("SourcedataB.csv", index=None)
			st.dataframe(df1)
            
			weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny',
                    'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
			temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
			play = ['No', 'No', 'Yes', 'Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

			le = preprocessing.LabelEncoder()
			weather_encoded = le.fit_transform(weather)
			st.write("weather:", weather_encoded)
			temp_encoded = le.fit_transform(temp)
			label = le.fit_transform(play)
			st.write("Temp:", temp_encoded)
			st.write("Play:", label)
			features = zip(weather_encoded, temp_encoded)
			features = list(features)
			st.write(features)
			model = GaussianNB()
			model.fit(features, label)
			predicted = model.predict([[0, 4]])
			st.write("Predicted Value:", predicted)

if choice == "DecisionTree":
    st.title("Telecharger votre fichier de donnees ici")
    file = st.file_uploader("Telechargement de donnees")
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age',
                'glucose', 'bp', 'pedigree', 'st', 'label']
    df.columns = feature_cols
 		
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("SourcedataD.csv", index=feature_cols)
        st.dataframe(df.head())
        numeric_cols = df.select_dtypes(exclude="object").columns.to_list()
        X = df[feature_cols]
        X = X.iloc[1:-1, :].values
        y = df.label
        y = y.iloc[1:-1].values
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        st.dataframe(df.shape)

if choice == "Cluster":
    st.title("Telecharger votre fichier de donnees ici")
    file = st.file_uploader("Telechargement de donnees")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("SourcedataC.csv", index=None)
        st.dataframe(df)
        X = df.iloc[:, [3, 4]].values
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        figc = plt.figure()
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
					s=100, c='red', label='Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
					s=100, c='blue', label='Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
					s=100, c='green', label='Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],
					s=100, c='cyan', label='Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],
					s=100, c='magenta', label='Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
					:, 1], s=300, c='yellow', label='Centroids')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        st.write(figc)
        pass

if choice == "PCA_Mall":
    st.title("Telecharger votre fichier de donnees ici")
    file = st.file_uploader("Telechargement de donnees")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("SourcedataP.csv", index=None)
        st.write("Dataset view")
        st.dataframe(df)

        X=df.iloc[:,1:-1].values
        Y=df.iloc[:,4].values

        X = np.array(X)
        X = X.reshape(-1,1)
        Y = np.array(Y)
        Y = Y.reshape(-1,1)
        
        le_X = MinMaxScaler()
        X_encoder =df.columns
        X_encoder= le_X.fit_transform(Y)

        
        Genre = preprocessing.LabelEncoder()
        Genre_encoder = Genre.fit_transform(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X_encoder, Y, test_size=0.25, random_state = 0)
        sc_X = StandardScaler()
        le_X.fit(X_train,Y_train)
        Y = le_X.fit_transform(Y)
        
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)


        classifierBayes=GaussianNB()
        classifierBayes.fit(X_train,Y_train)
        kpca=KernelPCA(n_components=1, kernel='rbf')
        X_trainPCA=kpca.fit_transform(X_train)
        
        classifierSVC=SVC(kernel='rbf', random_state=0)
        classifierSVC.fit(X_trainPCA,Y_train)

        Y_predSVC=classifierSVC.predict(X_test)
        Y_predBayes=classifierBayes.predict(X_test)
        accuracySVC=accuracy_score(Y_predSVC,Y_test)
        cmSVC=confusion_matrix(Y_test,Y_predSVC)
        st.write('Accuracy SVC')
        st.write(accuracySVC)
        st.write('Confusion matrix SVC')
        st.write(cmSVC)
        accuracyBayes=accuracy_score(Y_predBayes,Y_test)
        cmBayes=confusion_matrix(Y_test,Y_predBayes)
        st.write('Accuracy Bayes')
        st.write(accuracyBayes)
        st.write('Confusion matrix Bayes')
        st.write(cmBayes)
        st.write('With PCA')
        kpca=KernelPCA(n_components=1, kernel='rbf')
        X_trainPCA=kpca.fit_transform(X_train)
        X_testPCA=kpca.transform(X_test)
        classifierSVC.fit(X_trainPCA,Y_train)
        Y_predSVC=classifierSVC.predict(X_testPCA)
        Y_predBayes=classifierBayes.predict(X_testPCA)
        accuracySVC=accuracy_score(Y_predSVC,Y_test)
        cmSVC=confusion_matrix(Y_test,Y_predSVC)
        st.write('Accuracy SVC')
        st.write(accuracySVC)
        st.write('Confusion matrix SVC')
        st.write(cmSVC)
        accuracyBayes=accuracy_score(Y_predBayes,Y_test)
        cmBayes=confusion_matrix(Y_test,Y_predBayes)
        st.write('Accuracy Bayes')
        st.write(accuracyBayes)
        st.write('Confusion matrix Bayes')
        st.write(cmBayes)

        fig4=plt.figure()
        X_set, y_set = X_trainPCA, Y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifierSVC.predict(np.array([X1.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
              plt.scatter(X_set, y_set, c = ListedColormap(('red', 'green'))(i), label = j)
              plt.title('SVM (Training set)')
              plt.xlabel('Age')
              plt.ylabel('Estimated Salary')
              plt.legend()
              plt.show()
        st.write("SVM Training Set:", fig4)
        fig5 = plt.figure()
        X_set, y_set = X_testPCA, Y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifierSVC.predict(np.array([X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
              plt.scatter(X_set, y_set, c = ListedColormap(('red', 'green'))(i), label = j)
              plt.title('SVM (Test set)')
              plt.xlabel('Age')
              plt.ylabel('Estimated Salary')
              plt.legend()
              plt.show()
        st.write("SVM Test set :", fig5 )
pass
