import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb

# Title
st.title('Advanced Demography and Modeling App')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV or Text file", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            # If file is CSV, read normally
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                # Provide options for delimiter
                delimiter_option = st.sidebar.selectbox('Select the delimiter for your text file', options=['Comma (,)', 'Semicolon (;)', 'Tab (\\t)', 'Pipe (|)', 'Space', 'Other'])
                if delimiter_option == 'Comma (,)':
                    delimiter = ','
                elif delimiter_option == 'Semicolon (;)':
                    delimiter = ';'
                elif delimiter_option == 'Tab (\\t)':
                    delimiter = '\t'
                elif delimiter_option == 'Pipe (|)':
                    delimiter = '|'
                elif delimiter_option == 'Space':
                    delimiter = ' '
                elif delimiter_option == 'Other':
                    delimiter = st.sidebar.text_input('Enter the delimiter for your text file')
                    if not delimiter:
                        st.error('Please enter a delimiter.')
                        return None
                else:
                    delimiter = ','
                data = pd.read_csv(uploaded_file, delimiter=delimiter)
            else:
                st.error('Unsupported file type.')
                return None
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return None
    else:
        st.sidebar.info("Using an example demographic dataset.")
        data = pd.read_csv('https://raw.githubusercontent.com/datasets/population/master/data/population.csv')
        data = data[data['Year'] == 2016]
        data = data.sample(100, random_state=42)
    return data

# Get data
data = user_input_features()

if data is not None:
    # Display data
    st.subheader('Dataset')
    st.write('Data Dimension: {} rows and {} columns.'.format(data.shape[0], data.shape[1]))
    st.dataframe(data.head())

    # Select only numerical columns and categorical columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Exploratory Data Analysis
    st.subheader('Exploratory Data Analysis')
    if st.checkbox('Show distributions of numerical features'):
        for col in numerical_columns:
            fig = px.histogram(data, x=col, marginal='box', title=f'Distribution of {col}')
            st.plotly_chart(fig)

    if st.checkbox('Show box plots of numerical features'):
        for col in numerical_columns:
            fig = px.box(data, y=col, title=f'Box plot of {col}')
            st.plotly_chart(fig)

    if st.checkbox('Show correlation heatmap'):
        corr = data[numerical_columns].corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
        st.plotly_chart(fig)

    # Combine feature columns (numerical + categorical)
    feature_columns = numerical_columns + categorical_columns
    target_columns = data.columns.tolist()

    # Sidebar selection for feature and target
    st.subheader('Feature Selection')
    problem_type = st.selectbox('Select Problem Type', ('Regression', 'Classification'))
    target = st.selectbox('Select Target Variable', target_columns)
    features = st.multiselect('Select Feature Variables', [col for col in feature_columns if col != target])

    # Check if features are selected
    if st.button('Run Modeling'):
        if len(features) == 0:
            st.error('Please select at least one feature variable.')
        else:
            # Prepare features and target
            X = data[features]
            y = data[target]

            # For regression, ensure the target variable is numeric
            if problem_type == 'Regression':
                if not np.issubdtype(y.dtype, np.number):
                    st.error('For regression problems, the target variable must be numeric. Please select an appropriate target variable.')
                    st.stop()

                # Option to transform the target variable
                if st.checkbox('Apply transformation to the target variable'):
                    transformation_option = st.selectbox('Select transformation',
                                                         ('None', 'Logarithmic', 'Square Root', 'Box-Cox'))
                    if transformation_option == 'Logarithmic':
                        y = np.log1p(y)  # log(1 + y)
                    elif transformation_option == 'Square Root':
                        y = np.sqrt(y)
                    elif transformation_option == 'Box-Cox':
                        from scipy import stats
                        y, _ = stats.boxcox(y + 1e-6)  # adding a small value to avoid zero
                    else:
                        pass  # No transformation

            # Convert categorical variables to dummies (one-hot encoding)
            X = pd.get_dummies(X)
            if problem_type == 'Classification' and y.dtype == 'O':
                y, uniques = pd.factorize(y)

            # Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Ask user if they want to use cross-validation
            use_cv = st.checkbox('Use Cross-Validation')
            if use_cv:
                cv_folds = st.slider('Select number of CV folds', 3, 10, value=5)

            # Select model
            if problem_type == 'Regression':
                model_option = st.selectbox('Select Machine Learning Model', (
                    'Linear Regression', 'Random Forest Regressor', 'XGBoost Regressor'))
                if model_option == 'Linear Regression':
                    model = LinearRegression()
                elif model_option == 'Random Forest Regressor':
                    model = RandomForestRegressor()
                elif model_option == 'XGBoost Regressor':
                    model = xgb.XGBRegressor(objective='reg:squarederror')

                # Hyperparameter Tuning (optional)
                if st.checkbox('Perform Hyperparameter Tuning'):
                    st.write('Hyperparameter Tuning is selected.')
                    if model_option == 'Random Forest Regressor':
                        n_estimators = st.multiselect('Select n_estimators values', [50, 100, 150, 200, 250, 300], default=[100, 200])
                        max_depth = st.multiselect('Select max_depth values', [None, 5, 10, 15, 20], default=[None, 10])
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth
                        }
                    elif model_option == 'XGBoost Regressor':
                        n_estimators = st.multiselect('Select n_estimators values', [50, 100, 150, 200, 250, 300], default=[100, 200])
                        max_depth = st.multiselect('Select max_depth values', [3, 5, 7, 9], default=[3, 5])
                        learning_rate = st.multiselect('Select learning_rate values', [0.01, 0.05, 0.1, 0.2], default=[0.05, 0.1])
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate
                        }
                    else:
                        params = {}

                    if params:
                        grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
                        grid.fit(X_scaled, y)
                        model = grid.best_estimator_
                        st.write('Best Parameters:', grid.best_params_)
                    else:
                        model.fit(X_scaled, y)
                else:
                    model.fit(X_scaled, y)

                if use_cv:
                    scoring = 'neg_mean_squared_error'
                    scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring=scoring)
                    mse_scores = -scores  # since cross_val_score uses negative MSE
                    rmse_scores = np.sqrt(mse_scores)
                    st.write(f'Cross-Validation MSE: {mse_scores.mean():.2f} (+/- {mse_scores.std():.2f})')
                    st.write(f'Cross-Validation RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})')
                else:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    # Predict
                    y_pred = model.predict(X_test)
                    # Evaluation
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    st.write('Test Mean Squared Error (MSE): {:.2f}'.format(mse))
                    st.write('Test Root Mean Squared Error (RMSE): {:.2f}'.format(rmse))
                    # Plotting
                    fig = px.scatter(x=y_test, y=y_pred, labels={
                                     'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted')
                    st.plotly_chart(fig)

                # Display feature importances if available
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = X.columns
                    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
                    st.subheader('Feature Importances')
                    st.dataframe(importance_df)
                    fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importances')
                    st.plotly_chart(fig)

            else:
                # Classification
                model_option = st.selectbox('Select Machine Learning Model', (
                    'Logistic Regression', 'Random Forest Classifier', 'XGBoost Classifier'))
                if model_option == 'Logistic Regression':
                    model = LogisticRegression(max_iter=1000)
                elif model_option == 'Random Forest Classifier':
                    model = RandomForestClassifier()
                elif model_option == 'XGBoost Classifier':
                    model = xgb.XGBoostClassifier(use_label_encoder=False, eval_metric='mlogloss')

                # Hyperparameter Tuning (optional)
                if st.checkbox('Perform Hyperparameter Tuning'):
                    st.write('Hyperparameter Tuning is selected.')
                    if model_option == 'Random Forest Classifier':
                        n_estimators = st.multiselect('Select n_estimators values', [50, 100, 150, 200, 250, 300], default=[100, 200])
                        max_depth = st.multiselect('Select max_depth values', [None, 5, 10, 15, 20], default=[None, 10])
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth
                        }
                    elif model_option == 'XGBoost Classifier':
                        n_estimators = st.multiselect('Select n_estimators values', [50, 100, 150, 200, 250, 300], default=[100, 200])
                        max_depth = st.multiselect('Select max_depth values', [3, 5, 7, 9], default=[3, 5])
                        learning_rate = st.multiselect('Select learning_rate values', [0.01, 0.05, 0.1, 0.2], default=[0.05, 0.1])
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate
                        }
                    else:
                        params = {}

                    if params:
                        grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
                        grid.fit(X_scaled, y)
                        model = grid.best_estimator_
                        st.write('Best Parameters:', grid.best_params_)
                    else:
                        model.fit(X_scaled, y)
                else:
                    model.fit(X_scaled, y)

                if use_cv:
                    scoring = 'accuracy'
                    scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring=scoring)
                    st.write(f'Cross-Validation Accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)')
                else:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    # Predict
                    y_pred = model.predict(X_test)
                    # Evaluation
                    acc = accuracy_score(y_test, y_pred)
                    st.write('Test Accuracy: {:.2f}%'.format(acc * 100))
                    st.text('Classification Report:')
                    st.text(classification_report(y_test, y_pred))
                    # Plotting
                    fig = px.scatter(x=y_test, y=y_pred, labels={
                                     'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted')
                    st.plotly_chart(fig)

                # Display feature importances if available
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = X.columns
                    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
                    st.subheader('Feature Importances')
                    st.dataframe(importance_df)
                    fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importances')
                    st.plotly_chart(fig)
else:
    st.warning('Awaiting CSV or Text file to be uploaded.')
