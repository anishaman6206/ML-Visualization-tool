import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
import os
import plotly.express as px
import shap

# Import your models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import lightgbm as lgb




st.set_page_config(layout='wide',page_title='ML Viz Tools')
# Load preloaded datasets
classification_datasets = {
    'Iris': pd.read_csv('Iris.csv'),
    'Wine': pd.read_csv('winequality_white.csv'),
    'Diabetes': pd.read_csv('diabetes.csv'),
    'Breast Cancer': pd.read_csv('breast-cancer.csv')
}

continuous_datasets = {
    'Boston Housing': pd.read_csv('HousingData.csv'),
    
}

def get_classifier(model_name):
    if model_name == 'Logistic Regression':
        C = st.sidebar.number_input('C (Inverse Regularization Strength)', 0.01, 10.0, 1.0, step=0.01)
        max_iter = st.sidebar.slider('Maximum Iterations', 100, 500, 100, step=10)
        model = LogisticRegression(C=C, max_iter=max_iter)
        
    elif model_name == 'K-Nearest Neighbors':
        n_neighbors = st.sidebar.slider('Number of Neighbors (k)', 1, 20, 5)
        weights = st.sidebar.selectbox('Weights', ['uniform', 'distance'])
        metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'manhattan', 'minkowski'])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        
    elif model_name == 'Decision Tree':
        criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'])
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 10)
        splitter = st.sidebar.selectbox('Splitter', ['best', 'random'])
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, splitter=splitter)
        
    elif model_name == 'Random Forest':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 1)
        bootstrap = st.sidebar.selectbox('Bootstrap Samples', [True, False])
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        
    elif model_name == 'Support Vector Machine':
        C = st.sidebar.number_input('C (Regularization Parameter)', 0.01, 10.0, 1.0, step=0.01)
        kernel = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        gamma = st.sidebar.selectbox('Gamma', ['scale', 'auto'])
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        
    elif model_name == 'Naive Bayes':
        var_smoothing = st.sidebar.number_input('Var Smoothing', 1e-10, 1.0, 1e-9, format="%.10f")
        model = GaussianNB(var_smoothing=var_smoothing)
        
    elif model_name == 'Gradient Boosting (XGBoost)':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 10)
        model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        
    elif model_name == 'AdaBoost':
        n_estimators = st.sidebar.slider('Number of Trees', 50, 500, 50, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 1.0, step=0.01)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        
    elif model_name == 'LightGBM':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        max_depth = st.sidebar.slider('Max Depth', -1, 50, -1)
        model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        
    elif model_name == 'Extra Trees Classifier':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 10)
        criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'])
        model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
        
    else:
        st.warning('Please select a valid classifier.')
        model = None
    return model

def classification(model_choice,model,X,y):
    col1, col2 = st.columns(2)
    with col1:
      st.write('Shape of dataset:', X.shape)
    with col2:  
      st.write('Number of classes:', len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    k = st.sidebar.slider('k for Cross-Validation', 2, 10, 5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=k)
    cv_scores = [round(score, 2) for score in cv_scores]
    cv_mean = np.mean(cv_scores)
    f1 = f1_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    
    st.subheader('Model Performance')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('Accuracy', f"{accuracy * 100:.2f}%")
    col2.metric('Cross-Validation Mean Accuracy', f"{cv_mean * 100:.2f}%")
    col5.metric('F1 Score', f"{f1:.2f}")
    col3.metric('Precision', f"{precision:.2f}")
    col4.metric('Recall', f"{recall:.2f}")
    
    if model_choice == 'Logistic Regression':
        st.write("### Logistic Regression - Decision Boundary Visualization")
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        else:
            X_vis = X.values
            
        model.fit(X_vis, y)
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig1, ax1 = plt.subplots(figsize=(8,4))
        plt.figure(figsize=(8, 6))
        ax1.contourf(xx, yy, Z, alpha=0.8)
        ax1.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', marker='o')
        ax1.set_title('Decision Boundary')
        st.pyplot(fig1)
        
        col1, col2 = st.columns(2)

        with col1:
               
               st.subheader('0: Non-Diabetec, 1: Diabetec')
               fig, ax = plt.subplots(figsize=(8,4))
               class_counts = pd.Series(y).value_counts()
               ax.pie(class_counts, labels=class_counts.index, autopct="%0.01f%%")
               ax.set_title('Class Distribution')
               st.pyplot(fig)
    
        with col2:
                st.subheader('HeatMap')
                fig, ax = plt.subplots(figsize=(8,4))
                sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', ax=ax)
                st.pyplot(fig)

    elif model_choice == 'Random Forest':
        st.write("### Random Forest - Decision Tree Visualization")
        fig1, ax1 = plt.subplots()
        plt.figure(figsize=(20, 10))
        plot_tree(model.estimators_[0], filled=True, feature_names=X.columns, class_names=True, rounded=True, ax=ax1)
        st.pyplot(fig1)

        col1, col2 = st.columns(2)

        with col1:
               
            st.subheader('0: Non-Diabetec, 1: Diabetec')
            fig, ax = plt.subplots()
            class_counts = pd.Series(y).value_counts()
            ax.pie(class_counts, labels=class_counts.index, autopct="%0.01f%%")
            ax.set_title('Class Distribution')
            st.pyplot(fig)
      

            
    
        with col2:
            fig, ax = plt.subplots()
            st.subheader('HeatMap')
            sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', ax=ax)
            st.pyplot(fig)
    
    elif model_choice == 'SVM':
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)
        st.write("SVM does not inherently provide visualization capabilities like tree-based models.")


        col1, col2 = st.columns(2)
    
        with col2:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', ax=ax)
            st.pyplot(fig)

    elif model_choice == 'Decision Tree':
        st.write("### Decision Tree Visualization")
        fig1, ax1 = plt.subplots()
        plt.figure(figsize=(20, 10))
        plot_tree(model, filled=True, feature_names=X.columns, class_names=True, rounded=True, ax=ax1)
        st.pyplot(fig1)

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)        
# other plotting actions...
    



# Sidebar for Problem Type and Dataset Selection
st.sidebar.title("ML Model Visualizer")
st.sidebar.header('Choose Problem Type')
problem_type = st.sidebar.selectbox('Problem Type', ['Classification', 'Continuous'])

st.sidebar.header('Choose Dataset')
if problem_type == 'Classification':
    dataset_choice = st.sidebar.selectbox('Select Dataset', list(classification_datasets.keys()))
    data = classification_datasets[dataset_choice]
    
    st.write(f"### Top 5 values of {dataset_choice} Dataset")
    st.write(data.head())
    
    st.sidebar.header('Download Dataset')
    csv = data.to_csv(index=False)
    st.sidebar.download_button(label='Download Data', data=csv, file_name=f'{dataset_choice}.csv', mime='text/csv')
    
    st.sidebar.header('Select Model')
    model_choice = st.sidebar.selectbox('Choose Model', ['Logistic Regression',
        'K-Nearest Neighbors',
        'Decision Tree',
        'Random Forest',
        'Support Vector Machine',
        'Naive Bayes',
        'Gradient Boosting (XGBoost)',
        'AdaBoost',
        'LightGBM',
        'Extra Trees Classifier'])
    model = get_classifier(model_choice)
    if dataset_choice == 'Iris':
        data.drop(columns=['Id'],inplace=True)
        X = data.drop(columns=['Species'])
        data['Species'] = pd.Categorical(data['Species'])
        y = data['Species'].cat.codes
        
    elif dataset_choice == 'Diabetes':
        X = data.drop(columns=['Outcome'])
        y = data['Outcome']
    classification(model_choice,model,X,y)
    

