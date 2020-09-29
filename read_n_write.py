#!/usr/bin/env python3

from reader import Reader

import json
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def principal(df, n_clusters=5, dimensions=2):
    """A pipeline where first a k-means clustering is performed for the dataset. 
    Then the dimensionality of dataset is reduced by PCA. 
    Finally a 2-dimensional scatter plot is drawn."""

    # K-means clustering:
    X = df.to_numpy()
    clusters = KMeans(n_clusters)
    clusters.fit(X)

    # Principal component analysis:
    pca = PCA(dimensions)
    pca.fit(X)
    r = pca.explained_variance_ratio_
    print("\nExplained variance ratio by principal components:",r,"explaining a total",sum(r),"of the total variance")
    Z=pca.transform(X)
    
    # Scatter plot (first PC as x-axis, second PC as y-axis and third PC as point size (with some rescaling) and colours signifying clusters):
    labels = df.index.tolist()
    fig, ax = plt.subplots()
    ax.scatter(Z[:,0], Z[:,1], s=4*(Z[:,2]+4), c=clusters.labels_)
    for i, lbl in enumerate(labels):
        rnd = 1+np.random.rand()/10-0.05
        cp = 'black'
        if (lbl == 'finland'):
            cp = 'blue'
        ax.annotate(lbl, xy=(Z[i,0], Z[i,1]), xytext=(Z[i,0], rnd*Z[i,1]), color=cp)  
    plt.show()    

def transform(df):
    """This method is used to standardize data."""
    pt = PowerTransformer()
    result = pt.fit_transform(df)
    return result

def main():
    """This is a test program to give a 'proof-of-concept' for how to mine data from CIA factbook jsons."""

    with open("Miniprojekti/weekly_json/2020-05-04_factbook.json", "r", encoding='utf8', errors='ignore') as f:  
        factbook = json.load(f)["countries"]   # This creates a nested dictionary with all the data in a single .json

    r = Reader(factbook)     # A Reader object with a more accessible interface and unnecessary data filtered out.
    result = r.read_data([   # A query to create a DataFrame from the interesting stuff. Comment out stuff you don't need.
    
    # Geographic data 

        # "continent",
        # "area",
        # "irrigated",

    # Population data

        # "population",
        "children",
        "median_age",
        "population_growth",
        "birth_rate",
        "death_rate",
        # "migration",
        "infant_mortality",
        "life_expectancy",
        "fertility",
        # "literacy",
        # "lit_men",
        # "lit_women",

    # Economic data

        # "growth",
        # "gdp",
        "agriculture",
        "industry",
        "services",
        # "unemployment",
        # "poverty",
        "low_decile",
        "high_decile",
        # "revenues",
        # "expenditures",
        "public_debt",
        # "inflation",
        # "reserves",
        # "foreign_debt",

    # Military spending

        # "military",

    # Transnational issues

        # "refugees",
        # "internal_refugees"
    ]) 
     
    cc = pd.read_csv("Miniprojekti/country_codes.csv", sep=";", index_col="country")   # Read country codes from a manually created .csv file.
    result = pd.merge(result, cc, how="inner", right_index=True, left_index=True)      # Add geoarea code to DataFrame result.

    print("\nThe following were not found for Finland:",r.get_missing_data("finland")) # See which of the values were not in the factbook for finland.

    unwanted_data = ["world","european_union"]
    result = result.drop(unwanted_data)       # Get rid of unwanted data points. 
    nr = len(result.index)
    cropped_result = result.dropna()          # Get rid of data points, which contain NaN values.
    nc = len(cropped_result.index)
    print("\n",(nr-nc),"results were dropped out of",nr,"because of missing data for a total of",nc,"data points.")
    print("\n Countries included in the analysis are:\n", list(cropped_result.index))

    # Standardize data and perform PCA and k-means clustering. (This is the official playground!)
    transformed = pd.DataFrame(transform(cropped_result), columns=cropped_result.columns, index=cropped_result.index)
    principal(transformed, n_clusters=6, dimensions=5)        

if __name__ == "__main__":
    main()