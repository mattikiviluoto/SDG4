#!/usr/bin/env python3

from reader import Reader

import json
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


def principal(df, n_clusters=5, dimensions=3):
    """A pipeline where first a k-means clustering is performed for the dataset. 
    Then the dimensionality of dataset is reduced by PCA. 
    Finally a 3-dimensional scatter plot is drawn (with point size as the third dimension).
    : param df : a pandas DataFrame with standardized values for the dependent variables to be clustered,
    : param n_clusters : how many clusters to form,
    : param dimensions : number of principal components to consider via the PCA. 
    """
    # K-means clustering:
    X = df.to_numpy()
    clusters = KMeans(n_clusters)
    clusters.fit(X)

    # Principal component analysis:
    pca = PCA(dimensions)
    pca.fit(X)
    r = pca.explained_variance_ratio_
    print("\nExplained variance ratio by principal components:",r,"explaining",sum(r),"of the total variance")
    Z=pca.transform(X)
    
    # Scatter plot (first PC as x-axis, second PC as y-axis and third PC as point size (with some rescaling) and colours signifying clusters):
    labels = df.index.tolist()
    fig, ax = plt.subplots()
    ax.scatter(Z[:,0], Z[:,1], s=(Z[:,2]+5)**2, c=clusters.labels_)
    for i, lbl in enumerate(labels):
        rnd = 1+np.random.rand()/10-0.05
        cp = 'black'
        if (lbl == 'finland'):
            cp = 'blue'
        ax.annotate(lbl, xy=(Z[i,0], Z[i,1]), xytext=(Z[i,0], rnd*Z[i,1]), color=cp)  
    plt.title(f"k-means clustering, with {n_clusters} clusters")
    plt.text(-4.7, 3.6, "Point size as third principal component")
    plt.text(-4.7, 3.4, f"(explains {r[2]:.2f} of variance)")
    plt.xlabel(f"First principal component (explains {r[0]:.2f} of variance)")
    plt.ylabel(f"Second principal component (explains {r[1]:.2f} of variance)")
    plt.show()    

def transform(df):
    """This method is used to standardize data.
    : param df : a pandas DataFrame with values to transform,
    : return : a DataFrame with standardized values.
    """
    pt = PowerTransformer()
    result = pt.fit_transform(df)
    return result

def combine(df):
    """ This method combines the CIA and SVG data.
    : param df : a pandas DataFrame with CIA data,
    : return : a DataFrame with CIA and SVG data combined.
    """
    # Choose data
    y = pd.read_csv("data_ICT.csv") 
            
    # Drop countries from CIA dataframe that are not in SDG data
    X=df[df['geo_area_code'].isin(y['geoAreaCode'].to_numpy())]
    
    # Drop countries drom SDG dataframe that are not in CIA data
    y=y[y['geoAreaCode'].isin(X['geo_area_code'].to_numpy())]
    print("\nSVG data found for following countries:",list(y['Country']))
    
    # Save also the missing countries data from CIA data
    missing=df[~df['geo_area_code'].isin(y['geoAreaCode'].to_numpy())]
    
    # Sort X and y
    X=X.sort_values(by=['geo_area_code'])
    y=y.sort_values(by=['geoAreaCode'])

    # Choose only wanted values from X and y to linear regression
    y=y['Value']
    X=X.drop(['geo_area_code'], axis=1)
    missing=missing.drop(['geo_area_code'], axis=1)

    return (X, y, missing)

def linear_regression(X, y, missing):
    """This method performs Linear Regression and plots scatter plots to visualize dependencies.
    : param X : Independent variables,
    : param y : Dependent variable values used to train the model,
    : param missing : Dependent variable values to be predicted.
    : return : A pandas DataFrame with predicted values.
    """
    # Evaluate the model by cross-validation (using R-squared values as the metric).
    reg = LinearRegression()
    scores=cross_val_score(reg, X, y, cv=3, scoring='r2')
    print("\nScores of cross-validation (for linear regression):", scores)
    print("Mean of scores of cross-validation:", np.mean(scores))

    # Construct the linear regression model
    reg = LinearRegression().fit(X, y)
    
    # Predict the values of the missing data
    prediction=pd.DataFrame({'Country': missing.index, 'Value': reg.predict(missing)})
    pred=prediction.to_numpy()
    
    # Plot the real and predicted values of SDG data for each input variable
    y=y.to_numpy()
    k=0
    for col_name in X.columns:
        fig, ax = plt.subplots()
        ax.scatter(X.to_numpy()[:,k], y, label='Real values', c='b')
        ax.scatter(missing.to_numpy()[:,k], pred[:,1], label='Predicted values', c='orange')
        for i, txt in enumerate(missing.index):
            ax.annotate(txt, (missing.loc[txt][col_name], pred[i][1]), color='black')
        for i, txt in enumerate(X.index):
            ax.annotate(txt, (X.loc[txt][col_name], y[i]), color='black')
        k=k+1
        plt.xlabel(col_name)
        plt.ylabel("Means of proportions of youth and adults with various ICT-skills")
        plt.legend(loc='upper left')
        plt.show()
    
    return prediction

def rfr_optimizer(X, y):
    """This method creates a random grid of different combinations of hyperparameters and uses cross validation to pick the best one.
    : param X : Independent variables,
    : param y : Dependent variable values used to train the model,
    : return : Dictionary with optimal parameters (names of the parameters as keys and values as values).
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 19)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(50, 150, num = 11)]
    # Minimum number of samples required to split an internal node
    min_samples_split = [2,3,4]
    # Minimum number of samples required to be at a leaf node
    min_samples_leaf = [1,2,3]

    # Create the random grid from different hyperparameters.
    random_grid = {'n_estimators': n_estimators,                    
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}
    
    # Use the random grid to search for best hyperparameters for Random Forest Regressor
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, search across 100 random combinations, using all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)
    # Check which set of parameters yielded best results and return that
    rf_random.fit(X, y)
    best = rf_random.best_params_
    return best 

def random_forest_regressor(X, y, missing):
    """This method performs a Random Forest Regressor with optimized hyperparameters and plots scatter plots to visualize dependencies.
    : param X : Independent variables,
    : param y : Dependent variable values used to train the model,
    : param missing : Dependent variable values to be predicted.
    : return : A pandas DataFrame with predicted values.
    """
    # Optimise hyperparameters and evaluate the model by cross-validation (using R-squared values as the metric).
    print("\nOptimising the Random Forest Regressor. This will take up to 60 seconds. Please wait patiently...")
    params = rfr_optimizer(X, y)
    print("Best parameters found for Random Forest Regressor are:\n", params)
    rfr = RandomForestRegressor(
        n_estimators=params['n_estimators'], 
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features='sqrt',
        bootstrap=True,
        oob_score=True
        )
    scores=cross_val_score(rfr, X, y, cv=3, scoring='r2')
    print("\nScores of cross-validation (Random Forest Regressor):", scores)
    print("Mean of scores of cross-validation:", np.mean(scores))

    # Construct the Random Forest Classifier model
    rfr = rfr.fit(X, y) 

    # Predict the values of the missing data
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    prediction=pd.DataFrame({'Country': missing.index, 'Value': rfr.predict(missing)})
    pred=prediction.to_numpy()
    
    # Plot the real and predicted values of SDG data for each input variable
    y=y.to_numpy()
    k=0
    for col_name in X.columns:
        fig, ax = plt.subplots()
        ax.scatter(X.to_numpy()[:,k], y, label='Real values', c='b')
        ax.scatter(missing.to_numpy()[:,k], pred[:,1], label='Predicted values', c='orange')
        for i, txt in enumerate(missing.index):
            ax.annotate(txt, (missing.loc[txt][col_name], pred[i][1]), color='black')
        for i, txt in enumerate(X.index):
            ax.annotate(txt, (X.loc[txt][col_name], y[i]), color='black')
        k=k+1
        plt.xlabel(col_name)
        plt.ylabel("Means of proportions of youth and adults with various ICT-skills")
        plt.legend(loc='upper left')
        plt.show()

    return prediction

def main():
    """This is a test program to give a 'proof-of-concept' for how to mine and combine data from CIA factbook and SDG jsons."""

    with open("factbook.json", "r", encoding='utf8', errors='ignore') as f:  
        factbook = json.load(f)["countries"]   # This creates a nested dictionary with all the data in a single .json

    r = Reader(factbook)   # A Reader object with a more accessible interface and unnecessary data filtered out.
    data_query = [         # A query to create a DataFrame from the interesting stuff. Comment out stuff you don't need.
    
    # Geographic data 

        # "continent",
        # "area",
        # "irrigated",

    # Population data

        # "population",
        # "children",
        "median_age",
        "population_growth",
        "birth_rate",
        # "death_rate",
        # "migration",
        "infant_mortality",
        "life_expectancy",
        # "fertility",
        # "literacy",
        # "lit_men",
        # "lit_women",

    # Economic data

        # "growth",
        "gdp",
        "agriculture",
        # "industry",
        "services",
        # "unemployment",
        # "poverty",
        "low_decile",
        "high_decile",
        # "revenues",
        # "expenditures",
        # "public_debt",
        # "inflation",
        # "reserves",
        # "foreign_debt",

    # Military spending

        # "military",

    # Transnational issues

        # "refugees",
        # "internal_refugees"
    ]
    result = r.read_data(data_query)    # Pass the query to the Reader object
     
    cc = pd.read_csv("country_codes.csv", sep=";", index_col="country")   # Read country codes from a manually created .csv file.
    result = pd.merge(result, cc, how="inner", right_index=True, left_index=True)      # Add geoarea code to DataFrame result.

    print("\nThe following were not found for Finland:",r.get_missing_data("finland")) # See which values were not in the factbook for finland.

    unwanted_data = ["world", "european_union", "cameroon", "japan", "luxembourg", "zambia", "mongolia", "romania"]  
    result = result.drop(unwanted_data)         # Manually choose regions not to be included in the analysis (larger entities or outliers)
    nr = len(result.index)
    cropped_result = result.dropna()            # Get rid of data points, which contain NaN values.
    nc = len(cropped_result.index)
    print("\n",(nr-nc),"results were dropped out of",nr,"because of missing data in CIA factbook for a total of",nc,"data points.")
    print("\nCIA data with variables:",data_query,"\nfound for following countries:\n", list(cropped_result.index))
    print("\nOutliers or larger regions manually left out:",unwanted_data)

    # Combine CIA factbook data with SDG data to form independent and dependent variables, as well as data to be used in predictions
    ind, dep, miss = combine(cropped_result)

    # Construct and analyze the linear regression model
    lin_pred = linear_regression(ind, dep, miss)       
    lin_pred.set_index('Country', inplace=True)
    lin_pred.rename(columns = {'Value': 'LR'}, inplace=True)
    
    # Construct and analyze the Random Forest regression model
    rfr_pred = random_forest_regressor(ind, dep, miss)          
    rfr_pred.set_index('Country', inplace=True)
    rfr_pred.rename(columns = {'Value': 'RFR'}, inplace=True)
    
    # Print the predicted values from both models for comparison
    values = pd.merge(lin_pred, rfr_pred, how="inner", right_index=True, left_index=True)  
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("\nComparison of predicted values for missing countries by different models:")
    print(values)  

    # Standardize data and perform PCA and k-means clustering. 
    cropped_result = cropped_result.drop(['geo_area_code'], axis=1)   # Geo area codes are no longer needed.
    transformed = pd.DataFrame(transform(cropped_result), columns=cropped_result.columns, index=cropped_result.index)
    principal(transformed, n_clusters=3, dimensions=3)
            
if __name__ == "__main__":
    main()
