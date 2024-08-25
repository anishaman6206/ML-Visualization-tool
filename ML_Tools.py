import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.tree import plot_tree, export_graphviz
from sklearn.decomposition import PCA

import plotly.express as px


# Import your models

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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



def get_classifier(model_name):
    if model_name == 'Logistic Regression':
        C = st.sidebar.number_input('C (Inverse Regularization Strength)', 0.01, 10.0, 1.0, step=0.01)
        max_iter = st.sidebar.slider('Maximum Iterations', 100, 500, 100, step=10)
        solver = st.sidebar.selectbox('Solver', ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'])
        class_weight = st.sidebar.selectbox('Class Weight', [None, 'balanced'])
    
        # Initialize penalty and l1_ratio based on solver
        penalty = 'l2'  # Default penalty
        l1_ratio = None  # Default for solvers that don't use ElasticNet
    
        if solver == 'saga':
            penalty = st.sidebar.selectbox('Penalty', ['l2', 'l1', 'elasticnet',None])
            l1_ratio = st.sidebar.slider('l1_ratio', 0.01, 1.0, 0.5, step=0.01)
        elif solver == 'lbfgs' or solver == 'sag' or solver=='newton-cg':
            penalty = st.sidebar.selectbox('Penalty', ['l2',None])

        else:
            penalty = st.sidebar.selectbox('Penalty', ['l2', 'l1'])
    
        # Create Logistic Regression model
        if solver == 'saga' and penalty == 'elasticnet':
            model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, penalty=penalty, l1_ratio=l1_ratio, class_weight=class_weight)
        else:
            model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, penalty=penalty, class_weight=class_weight)
        
    

        
    elif model_name == 'K-Nearest Neighbors':
        n_neighbors = st.sidebar.slider('Number of Neighbors (k)', 1, 20, 5)
        weights = st.sidebar.selectbox('Weights', ['uniform', 'distance'])
        metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'manhattan', 'minkowski'])
        p = st.sidebar.slider('Power Parameter for Minkowski Metric (p)', 1, 5, 2)
        algorithm = st.sidebar.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf_size = st.sidebar.slider('Leaf Size for Tree Algorithm', 10, 100, 30, step=5)
    
        model = KNeighborsClassifier(
        n_neighbors=n_neighbors, 
        weights=weights, 
        metric=metric, 
        p=p, 
        algorithm=algorithm, 
        leaf_size=leaf_size
    )

        
    elif model_name == 'Decision Tree':
        criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'])
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 2)
        splitter = st.sidebar.selectbox('Splitter', ['best', 'random'])
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('Min Samples per Leaf', 1, 20, 1)
        max_features = st.sidebar.selectbox('Max Features', ['sqrt', 'log2', None])
        
        model = DecisionTreeClassifier(
        criterion=criterion, 
        max_depth=max_depth, 
        splitter=splitter, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features
    )

        
    elif model_name == 'Random Forest':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'])
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 2)
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('Min Samples per Leaf', 1, 20, 1)
        max_features = st.sidebar.selectbox('Max Features', ['sqrt', 'log2', None])
        bootstrap = st.sidebar.selectbox('Bootstrap Samples', [True, False])
        
        model = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion=criterion, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features, 
        bootstrap=bootstrap
    )
        
   
    elif model_name == 'Support Vector Machine':
        C = st.sidebar.number_input('C (Regularization Parameter)', 0.01, 10.0, 1.0, step=0.01)
        kernel = st.sidebar.selectbox('Choose Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

        if kernel == 'poly':
            degree = st.sidebar.slider('Degree of Polynomial Kernel', 2, 10, 3)
            model = SVC(kernel='poly', degree=degree)
        elif kernel == 'rbf':
            gamma = st.sidebar.slider('Gamma for RBF Kernel', 0.1, 1.0, 0.5)
            model = SVC(kernel='rbf', gamma=gamma)
        else:
            model = SVC(kernel=kernel)
        
    elif model_name == 'Naive Bayes':
        var_smoothing = st.sidebar.number_input('Var Smoothing', 1e-10, 1.0, 1e-9, format="%.10f")
        model = GaussianNB(var_smoothing=var_smoothing)
        
    

    elif model_name == 'Gradient Boosting (XGBoost)':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 10)
        min_child_weight = st.sidebar.slider('Min Child Weight', 1, 10, 1)
        subsample = st.sidebar.number_input('Subsample', 0.1, 1.0, 1.0, step=0.1)
        colsample_bytree = st.sidebar.number_input('Colsample Bytree', 0.1, 1.0, 1.0, step=0.1)
        gamma = st.sidebar.number_input('Gamma', 0.0, 10.0, 0.0, step=0.1)
        
        model = xgb.XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma
    )
        
    elif model_name == 'AdaBoost':
        n_estimators = st.sidebar.slider('Number of Trees', 50, 500, 50, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 1.0, step=0.01)
        algorithm = st.sidebar.selectbox('Algorithm', ['SAMME', 'SAMME.R'])
        
        model = AdaBoostClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate,
        algorithm=algorithm
    )

        
    elif model_name == 'LightGBM':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        max_depth = st.sidebar.slider('Max Depth', -1, 50, -1)
        num_leaves = st.sidebar.slider('Num Leaves', 2, 256, 31)
        min_child_samples = st.sidebar.slider('Min Child Samples', 1, 100, 20)
        subsample = st.sidebar.number_input('Subsample', 0.1, 1.0, 1.0, step=0.1)
        colsample_bytree = st.sidebar.number_input('Colsample Bytree', 0.1, 1.0, 1.0, step=0.1)
        
        model = lgb.LGBMClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    elif model_name == 'Extra Trees Classifier':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 2)
        criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('Min Samples per Leaf', 1, 20, 1)
        max_features = st.sidebar.selectbox('Max Features', ['sqrt', 'log2', None])
        
        model = ExtraTreesClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )

        
    else:
        st.warning('Please select a valid classifier.')
        model = None
    return model

def get_regressor(model_name):
    if model_name == 'Linear Regression':
        model = LinearRegression()
    
    elif model_name == 'Ridge Regression':
        alpha = st.sidebar.slider('Alpha', 0.01, 10.0, 1.0, step=0.01)
        model = Ridge(alpha=alpha)
    
    elif model_name == 'Lasso Regression':
        alpha = st.sidebar.slider('Alpha', 0.01, 10.0, 1.0, step=0.01)
        model = Lasso(alpha=alpha)
    elif model_name == 'ElasticNet':
        alpha = st.sidebar.slider('Alpha', 0.01, 10.0, 1.0, step=0.01)
        model = ElasticNet(alpha=alpha)     

    elif model_name == 'Polynomial Regression':
        
        
        # Degree of the polynomial
        degree = st.sidebar.slider('Polynomial Degree', 1, 10, 2, step=1)
        
        # Regularization for Linear Regression
        regularization = st.sidebar.selectbox('Regularization', ['None', 'Ridge', 'Lasso'])
        if regularization == 'None':
            model_type = 'Linear Regression'
        elif regularization == 'Ridge':
            alpha = st.sidebar.slider('Alpha (Ridge)', 0.01, 10.0, 1.0, step=0.01)
            model_type = 'Ridge Regression'
        elif regularization == 'Lasso':
            alpha = st.sidebar.slider('Alpha (Lasso)', 0.01, 10.0, 1.0, step=0.01)
            model_type = 'Lasso Regression'
    
        # Polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        
        # Choose the linear model
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'Ridge Regression':
            model = Ridge(alpha=alpha)
        elif model_type == 'Lasso Regression':
            model = Lasso(alpha=alpha)    
        
        elif model_name == 'ElasticNet':
            alpha = st.sidebar.slider('Alpha', 0.01, 10.0, 1.0, step=0.01)
            l1_ratio = st.sidebar.slider('L1 Ratio', 0.0, 1.0, 0.5)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    
    elif model_name == 'Decision Tree Regressor':
        criterion = st.sidebar.selectbox('Criterion', ['absolute_error', 'squared_error', 'friedman_mse', 'poisson'])
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 2)
        max_features = st.sidebar.selectbox('Max Features', ['sqrt', 'log2', None])
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('Min Samples per Leaf', 1, 20, 1)
        splitter = st.sidebar.selectbox('Splitter', ['best', 'random'])
        model = DecisionTreeRegressor(criterion=criterion, max_features=max_features,max_depth=max_depth, splitter=splitter,min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf)
    
    elif model_name == 'Random Forest Regressor':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        max_features = st.sidebar.selectbox('Max Features', ['sqrt', 'log2', None])
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 2)
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('Min Samples per Leaf', 1, 20, 1)
        bootstrap = st.sidebar.selectbox('Bootstrap Samples', [True, False])

        
        model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap
    )


    elif model_name == 'K-Nearest Neighbors Regressor':
        n_neighbors = st.sidebar.slider('Number of Neighbors (k)', 1, 20, 5)
        weights = st.sidebar.selectbox('Weights', ['uniform', 'distance'])
        algorithm = st.sidebar.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        p = st.sidebar.slider('Power Parameter for Minkowski Distance', 1, 10, 2)
        
        model = KNeighborsRegressor(
        n_neighbors=n_neighbors, 
        weights=weights, 
        algorithm=algorithm,
        p=p
    )

        
    
    elif model_name == 'Support Vector Regressor (SVR)':
       C = st.sidebar.number_input('C (Regularization Parameter)', 0.01, 10.0, 1.0, step=0.01)
       epsilon = st.sidebar.number_input('Epsilon', 0.01, 1.0, 0.1, step=0.01)
       kernel = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
       degree = st.sidebar.slider('Degree for Polynomial Kernel', 1, 5, 3)
       gamma = st.sidebar.selectbox('Gamma', ['scale', 'auto'])
       
       model = SVR(
        C=C, 
        epsilon=epsilon, 
        kernel=kernel,
        degree=degree,
        gamma=gamma
    )

    
    elif model_name == 'Gradient Boosting Regressor (XGBoost)':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 10)
        
        subsample = st.sidebar.number_input('Subsample', 0.1, 1.0, 1.0, step=0.1)
        
       
        
        model = GradientBoostingRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth,
        subsample=subsample,
        
        
    )

    
    elif model_name == 'AdaBoost Regressor':
        n_estimators = st.sidebar.slider('Number of Trees', 50, 500, 50, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 1.0, step=0.01)
        loss = st.sidebar.selectbox('Loss Function', ['linear', 'square', 'exponential'])
    
        model = AdaBoostRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate,
        loss=loss
    )

    elif model_name == 'LightGBM Regressor':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        max_depth = st.sidebar.slider('Max Depth', -1, 50, -1)
        num_leaves = st.sidebar.slider('Num Leaves', 2, 256, 31)
        min_child_samples = st.sidebar.slider('Min Child Samples', 1, 100, 20)
        subsample = st.sidebar.number_input('Subsample', 0.1, 1.0, 1.0, step=0.1)
        colsample_bytree = st.sidebar.number_input('Colsample Bytree', 0.1, 1.0, 1.0, step=0.1)
        
        model = lgb.LGBMRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    elif model_name == 'XGBoost Regressor':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        learning_rate = st.sidebar.number_input('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 10)
        min_child_weight = st.sidebar.slider('Min Child Weight', 1, 10, 1)
        subsample = st.sidebar.number_input('Subsample', 0.1, 1.0, 1.0, step=0.1)
        colsample_bytree = st.sidebar.number_input('Colsample Bytree', 0.1, 1.0, 1.0, step=0.1)
        gamma = st.sidebar.number_input('Gamma', 0.0, 10.0, 0.0, step=0.1)
        
        model = xgb.XGBRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma
    )

    
    elif model_name == 'Extra Trees Regressor':
        n_estimators = st.sidebar.slider('Number of Trees', 100, 500, 100, step=50)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 2)
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('Min Samples per Leaf', 1, 20, 1)
        max_features = st.sidebar.selectbox('Max Features', ['sqrt', 'log2', None])
        criterion = st.sidebar.selectbox('Criterion', ['absolute_error', 'squared_error', 'friedman_mse', 'poisson'])
        
        model = ExtraTreesRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion
    )

    
    else:
        st.warning('Please select a valid regressor.')
        model = None
    
    return model

def classification(data,label_mapping,model_choice,model,X,y):
    col1, col2 = st.columns(2)
    with col1:
      st.write('Shape of dataset:', X.shape)
    with col2:  
      st.write('Number of classes:', len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Standardize features for better visualization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

  
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
        h=0.02  # Step size in the meshgrid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0), np.arange(y_min, y_max, 1.0))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig1, ax1 = plt.subplots(figsize=(8,4))
        contour = ax1.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')  # Choose an appropriate colormap
        scatter = ax1.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', cmap='coolwarm', marker='o')
        
        # Get unique class labels
        unique_classes = np.unique(y)
        
        # Map labels using label_mapping
        labels = [label_mapping.get(i, str(i)) for i in unique_classes]
        
        # Create legend handles with colors corresponding to scatter
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(cls)), markersize=5, label=label) for cls, label in zip(unique_classes, labels)]
        ax1.legend(handles=handles, loc='best', title='Classes')
        ax1.set_title('Decision Boundary')
        st.pyplot(fig1)
    
    elif model_choice == 'K-Nearest Neighbors':
        st.write("### K-Nearest Neighbors - Decision Boundary Visualization")
        
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        else:
            X_vis = X.values
    
        model.fit(X_vis, y)
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0), np.arange(y_min, y_max, 1.0))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        contour = ax1.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        scatter = ax1.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    
        # Get unique class labels
        unique_classes = np.unique(y)
        
        # Map labels using label_mapping
        labels = [label_mapping.get(i, str(i)) for i in unique_classes]
        
        # Create legend handles with colors corresponding to scatter
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(cls)), markersize=5, label=label) for cls, label in zip(unique_classes, labels)]
        
        ax1.legend(handles=handles, loc='best', title='Classes')
        ax1.set_title(f'K-Nearest Neighbors Decision Boundary (k={model.n_neighbors})')
        st.pyplot(fig1)
    elif model_choice == 'Decision Tree':
        st.write("### Decision Tree - Tree Visualization")
    
        fig1, ax1 = plt.subplots()
        plt.figure(figsize=(20, 10))
        plot_tree(model, filled=True, feature_names=X.columns, class_names=True, rounded=True, ax=ax1)
        st.pyplot(fig1)

    elif model_choice == 'Random Forest':
        st.write("### Random Forest - Decision Tree Visualization")
        fig1, ax1 = plt.subplots()
        plt.figure(figsize=(20, 10))
        plot_tree(model.estimators_[0], filled=True, feature_names=X.columns, class_names=True, rounded=True, ax=ax1)
        st.pyplot(fig1)


      
    
    elif model_choice == 'Support Vector Machine':
        st.write("### Support Vector Machine - Decision Boundary Visualization")
        
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        else:
            X_vis = X.values
    
        model.fit(X_vis, y)
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0), np.arange(y_min, y_max, 1.0))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        contour = ax1.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')  # Choose an appropriate colormap
        scatter = ax1.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    
        # Plot support vectors
        ax1.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                    facecolors='none', edgecolors='k', marker='o', label='Support Vectors')
    
        # Get unique class labels
        unique_classes = np.unique(y)
        
        # Map labels using label_mapping
        labels = [label_mapping.get(i, str(i)) for i in unique_classes]
        
        # Create legend handles with colors corresponding to scatter
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(cls)), markersize=5, label=label) for cls, label in zip(unique_classes, labels)]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=10, label='Support Vectors'))
        
        ax1.legend(handles=handles, loc='best', title='Classes')
        ax1.set_title('SVM Decision Boundary')
        st.pyplot(fig1)

        
  

    elif model_choice == 'Naive Bayes':
        st.write("### Naive Bayes - Decision Boundary Visualization")
        
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        else:
            X_vis = X.values
    
        model.fit(X_vis, y)
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0), np.arange(y_min, y_max, 1.0))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        contour = ax1.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        scatter = ax1.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    
        # Get unique class labels
        unique_classes = np.unique(y)
        
        # Map labels using label_mapping
        labels = [label_mapping.get(i, str(i)) for i in unique_classes]
        
        # Create legend handles with colors corresponding to scatter
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(cls)), markersize=5, label=label) for cls, label in zip(unique_classes, labels)]
        
        ax1.legend(handles=handles, loc='best', title='Classes')
        ax1.set_title('Naive Bayes Decision Boundary')
        st.pyplot(fig1)

        
      


    elif model_choice in ['Gradient Boosting (XGBoost)', 'AdaBoost', 'LightGBM', 'Extra Trees Classifier']:

        if model_choice in ['Extra Trees Classifier']:
            st.write(f"### {model_choice} - Decision Tree Visualization")
            fig2, ax2 = plt.subplots(figsize=(20, 10))
            plot_tree(model.estimators_[0], filled=True, feature_names=X.columns, class_names=True, rounded=True, ax=ax2)
            st.pyplot(fig2)
        
            importance = model.feature_importances_
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.barplot(x=importance, y=X.columns, ax=ax1)
            ax1.set_title(f'{model_choice} - Feature Importance')
            st.pyplot(fig1)
             
        else:
            
            importance = model.feature_importances_
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.barplot(x=importance, y=X.columns, ax=ax1)
            ax1.set_title(f'{model_choice} - Feature Importance')
            st.pyplot(fig1)   
    col1, col2 = st.columns(2) 
# other plotting actions...
    with col1:
        st.subheader('Output Column Distribution')
        fig, ax = plt.subplots(figsize=(8, 4))
        class_counts = pd.Series(y).value_counts()
        
        labels = [label_mapping.get(i, str(i)) for i in class_counts.index]
        ax.pie(class_counts, labels=labels, autopct="%0.01f%%")
        ax.set_title('Class Distribution')
        st.pyplot(fig)

    with col2:
        st.subheader('Heatmap of Confusion Matrix')
        fig, ax = plt.subplots(figsize=(12, 10))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[label_mapping.get(i, str(i)) for i in class_counts.index], yticklabels=[label_mapping.get(i, str(i)) for i in class_counts.index], ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
# Correlation Plot
    st.subheader('Correlation Plot')
    fig, ax = plt.subplots(figsize=(8, 4))
    corr = X.corr()
    sns.heatmap(corr, annot=True, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

def regression(data,model_choice, model, X, y):
    col1, col2 = st.columns(2)
    with col1:
        st.write('Shape of dataset:', X.shape)
    with col2:
        st.write('Number of features:', X.shape[1])
            # Plot histogram for the target variable
  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    k = st.sidebar.slider('k for Cross-Validation', 2, 10, 5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=k, scoring='neg_mean_squared_error')
    cv_scores = np.sqrt(-cv_scores)
    cv_mean = np.mean(cv_scores)
    
    st.subheader('Model Performance')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('RMSE', f"{rmse:.2f}")
    col2.metric('MAE', f"{mae:.2f}")
    col3.metric(label="$R^2$ Score", value=f"{r2:.2f}")
    col4.metric('Cross-Validation Mean RMSE', f"{cv_mean:.2f}")


    
    if model_choice in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        st.write(f"### {model_choice} - Feature Coefficients")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        coef = pd.Series(model.coef_, index=X.columns)
        coef.plot(kind='bar', ax=ax3)
        ax3.set_ylabel('Features')
        ax3.set_title(f'{model_choice} - Coefficients')
        st.pyplot(fig3)

    if model_choice == 'Polynomial Regression':
        st.write(f"### {model_choice} - Regression Curve Visualization")
    
        # Create polynomial features
        # Create polynomial features
        degree = st.sidebar.slider('Degree', 2, 10, 2)
        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)

    
        # Apply PCA to reduce features to 2D for visualization
        # Apply PCA to the polynomial features
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_poly_train)
        X_test_pca = pca.transform(X_poly_test)

        # Fit the model on the PCA-transformed polynomial features
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        predictions = model.predict(X_test_pca)


    
       
        # Sort for smooth plotting
        sorted_idx = np.argsort(X_test_pca[:, 0])
        X_test_sorted = X_test_pca[sorted_idx]
        predictions_sorted = predictions[sorted_idx]
        
        # Plot the regression curve
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        ax1.scatter(X_test_pca[:, 0], y_test, color='blue', label='Data Points')
        ax1.plot(X_test_sorted[:, 0], predictions_sorted, color='red', label='Regression Curve')
        ax1.set_title(f'{model_choice} - Actual vs Predicted with PCA')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Output')
        ax1.legend()
        st.pyplot(fig1)


    elif model_choice in ['Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor (XGBoost)', 'AdaBoost Regressor', 'Extra Trees Regressor']:
        st.write(f"### {model_choice} - Feature Importance")
        importance = model.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=importance, y=X.columns, ax=ax)
        ax.set_title(f'{model_choice} - Feature Importance')
        ax.set_ylabel('Features')
        st.pyplot(fig)
        
        if model_choice == 'Decision Tree Regressor':
            st.write(f"### Decision Tree - Tree Visualization")
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(model, filled=True, feature_names=X.columns, ax=ax)
            st.pyplot(fig)
        
        elif model_choice == 'Random Forest Regressor':
            st.write(f"### Random Forest - Tree Visualization")
            estimator = model.estimators_[0]  # Visualize the first tree
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.set_ylabel('Feature')
            plot_tree(estimator, filled=True, feature_names=X.columns, ax=ax)
            ax.legend()
            st.pyplot(fig)
        
        
        elif model_choice == 'AdaBoost Regressor':
            st.write(f"### AdaBoost - Tree Visualization")
            estimator = model.estimators_[0]  # Visualize the first tree
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(estimator, filled=True, feature_names=X.columns, ax=ax)
            st.pyplot(fig)
        
        elif model_choice == 'Extra Trees Regressor':
            st.write(f"### Extra Trees - Tree Visualization")
            estimator = model.estimators_[0]  # Visualize the first tree
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(estimator, filled=True, feature_names=X.columns, ax=ax)
            st.pyplot(fig)

    
    elif model_choice in ['Support Vector Regressor (SVR)']:
        st.write("### Support Vector Regressor - SVR")
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        else:
            X_vis = X.values

        model.fit(X_vis, y)
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0), np.arange(y_min, y_max, 1.0))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.contourf(xx, yy, Z, alpha=0.8)
        ax1.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', marker='o')
        ax1.set_title('SVR Decision Boundary')
        st.pyplot(fig1)

   
    col1, col2 = st.columns(2) 
    with col1:       

        st.write("### Regression Plot")        
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        ax1.scatter(y_test, predictions, color='blue')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax1.set_title(f'{model_choice} - Actual vs Predicted')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        st.pyplot(fig1)
    # Additional visualizations for feature importance, residuals, etc.
    with col2:
        st.write("### Residuals Plot")
        residuals = y_test - predictions
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.scatter(predictions, residuals, color='blue')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('Residuals vs Predicted')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals')
        st.pyplot(fig2)        
    
    st.write('### Target Variable Distribution')
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    sns.histplot(y, kde=True, ax=ax_hist, bins=30, color='skyblue')
    ax_hist.set_title('Distribution of Target Variable')
    ax_hist.set_xlabel('Target')
    ax_hist.set_ylabel('Frequency')
    st.pyplot(fig_hist)
     
    st.subheader('Correlation Plot')
    fig, ax = plt.subplots(figsize=(8, 4))
    corr = X.corr()
    sns.heatmap(corr, annot=True, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig) 

st.set_page_config(layout='wide',page_title='ML Viz Tools')
# Load preloaded datasets
classification_datasets = {
    'Iris': pd.read_csv('Iris.csv'),
    'Diabetes': pd.read_csv('diabetes.csv'),
    'Breast Cancer': pd.read_csv('breast-cancer.csv'),
    'Wine': pd.read_csv('winequality_white.csv')

}

regression_datasets = {
    'Boston Housing': pd.read_csv('HousingData.csv'),
    'Automobile': pd.read_csv('automobile.csv', index_col=0)
    
}    

# Sidebar for Problem Type and Dataset Selection
st.header("Choose different Datasets, Hypertune Models and Visualize the Changes")
st.sidebar.title("ML Model Visualizer")
st.sidebar.header('Choose Problem Type')
problem_type = st.sidebar.selectbox('Problem Type', ['Classification', 'Regression'])

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
    # Documentation links
    documentation_links = {
    'Logistic Regression': 'https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression',
    'K-Nearest Neighbors': 'https://scikit-learn.org/stable/modules/neighbors.html#classification',
    'Decision Tree': 'https://scikit-learn.org/stable/modules/tree.html#classification',
    'Random Forest': 'https://scikit-learn.org/stable/modules/ensemble.html#forest',
    'Support Vector Machine': 'https://scikit-learn.org/stable/modules/svm.html#classification',
    'Naive Bayes': 'https://scikit-learn.org/stable/modules/naive_bayes.html',
    'Gradient Boosting (XGBoost)': 'https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting',
    'AdaBoost': 'https://scikit-learn.org/stable/modules/ensemble.html#adaboost',
    'LightGBM': 'https://lightgbm.readthedocs.io/en/latest/Parameters.html#classification',
    'Extra Trees Classifier': 'https://scikit-learn.org/stable/modules/ensemble.html#forest'
}

# Display documentation link
    if st.sidebar.button(f'View Documentation of {model_choice}'):
        if model_choice in documentation_links:
            st.sidebar.markdown(f"[Documentation for {model_choice}]({documentation_links[model_choice]})")
        else:
            st.sidebar.markdown("No documentation available for the selected model.")
    model = get_classifier(model_choice)
    if dataset_choice == 'Iris':
        data.drop(columns=['Id'],inplace=True)
        X = data.drop(columns=['Species'])
        data['Species'] = pd.Categorical(data['Species'])
        y = data['Species'].cat.codes
        label_mapping = {0: 'setosa', 1: 'versicolor', 2:'virginica'}
        
    elif dataset_choice == 'Diabetes':

        X = data.drop(columns=['Outcome'])
        y = data['Outcome']
        # Dictionary for label mapping
        label_mapping = {0: 'Non-Diabetic', 1: 'Diabetic'}

    elif dataset_choice == 'Breast Cancer':
        data.drop(columns=['id','concave points_mean'],inplace=True)
        # Convert all float64 columns to float32
        data = data.apply(lambda col: col.astype(np.float32) if col.dtype == np.float64 else col)
        
        X = data.drop('diagnosis', axis=1)
        data['diagnosis'] = data['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)
        y = data['diagnosis']
        label_mapping = {0: 'Benign', 1: 'Malignant'} 
        

    elif dataset_choice=='Wine':
        y = data['quality']
        X = data.drop('quality', axis=1)
        label_mapping = {8: 'Quality: 8 ', 3: 'Quality: 3',4: 'Quality: 4 ', 5: 'Quality: 5',6: 'Quality: 6 ', 7: 'Quality: 7',9: 'Quality: 9'}    


    
    
    classification(data,label_mapping,model_choice,model,X,y)

    
  

    

if problem_type == 'Regression':
    dataset_choice = st.sidebar.selectbox('Select Dataset', list(regression_datasets.keys()))
    data = regression_datasets[dataset_choice]


    
    st.write(f"### Top 5 values of {dataset_choice} Dataset")
    st.write(data.head())
    
    st.sidebar.header('Download Dataset')
    csv = data.to_csv(index=False)
    st.sidebar.download_button(label='Download Data', data=csv, file_name=f'{dataset_choice}.csv', mime='text/csv')
    
    st.sidebar.header('Select Model')
    model_choice = st.sidebar.selectbox('Choose Regression Model', [
    'Linear Regression',
    'Ridge Regression',
    'Lasso Regression',
    'ElasticNet',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'Support Vector Regressor (SVR)',
    'K-Nearest Neighbors Regressor',
    'Gradient Boosting Regressor (XGBoost)',
    'AdaBoost Regressor',
    'LightGBM Regressor',
    'Extra Trees Regressor'
])
# Model documentation links
    documentation_links = {
    'Linear Regression': 'https://scikit-learn.org/stable/modules/linear_model.html#linear-regression',
    'Ridge Regression': 'https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression',
    'Lasso Regression': 'https://scikit-learn.org/stable/modules/linear_model.html#lasso',
    'ElasticNet': 'https://scikit-learn.org/stable/modules/linear_model.html#elastic-net',
    'Decision Tree Regressor': 'https://scikit-learn.org/stable/modules/tree.html#regression',
    'Random Forest Regressor': 'https://scikit-learn.org/stable/modules/ensemble.html#forest',
    'Support Vector Regressor (SVR)': 'https://scikit-learn.org/stable/modules/svm.html#regression',
    'K-Nearest Neighbors Regressor': 'https://scikit-learn.org/stable/modules/neighbors.html#regression',
    'Gradient Boosting Regressor (XGBoost)': 'https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting',
    'AdaBoost Regressor': 'https://scikit-learn.org/stable/modules/ensemble.html#adaboost',
    'LightGBM Regressor': 'https://lightgbm.readthedocs.io/en/latest/Parameters.html#regression',
    'Extra Trees Regressor': 'https://scikit-learn.org/stable/modules/ensemble.html#forest'
}

# Display documentation link
    if st.sidebar.button(f'View Documentation of {model_choice}'):
        if model_choice in documentation_links:
            st.sidebar.markdown(f"[Documentation for {model_choice}]({documentation_links[model_choice]})")
        else:
            st.sidebar.markdown("No documentation available for the selected model.")
    model = get_regressor(model_choice)
    if dataset_choice == 'Boston Housing':
        data['CRIM'].fillna(data['CRIM'].mean(), inplace=True)
        data['ZN'].fillna(data['ZN'].mean(), inplace=True)
        data['INDUS'].fillna(data['INDUS'].mean(), inplace=True)
        data['CHAS'].fillna(data['CHAS'].mode()[0], inplace=True)  
        data['AGE'].fillna(data['AGE'].mean(), inplace=True)
        data['LSTAT'].fillna(data['LSTAT'].mean(), inplace=True)
        X = data.drop('MEDV', axis=1)  # Features
        y = data['MEDV']
     
    elif dataset_choice == 'Automobile':
     
     data = data.replace('?', pd.NA)
     data = data.astype(object).where(pd.notnull(data), np.nan)
 
    # Identify categorical columns
    # List of categorical and numerical columns
     categorical_columns = [
    'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
    'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system'
]

# Fill missing values in each categorical column with the mode (most frequent value)
     for col in categorical_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

# List of numerical columns
     numerical_columns = [
    'normalized-losses', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
    'city-mpg', 'highway-mpg', 'price'
]
    
    # Handle missing values in numerical columns
     imputer = SimpleImputer(strategy='mean')
     data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
    
    

# Convert categorical columns to numeric using one-hot encoding
     encoder = OneHotEncoder(drop='first', sparse_output=False)
     encoded_features = encoder.fit_transform(data[categorical_columns])
     encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
     
    
    # Combine encoded features with numerical columns
     data = pd.concat([encoded_df, data[numerical_columns]], axis=1)
     st.write('Top 5 Data Values after encoding categorical column')
     st.write(data.head())
  
    
    
    # Prepare features and target variable
     X = data.drop('price', axis=1)  # Features
     y = data['price']     
    regression(data,model_choice,model,X,y)    

