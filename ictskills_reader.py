import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

# Read the data
data = gpd.read_file("ictskills.geojson")

# Take only the wanted values from the dataframe
data_all = data.loc[0:len(data),['latest_value',  'geoAreaName',  'geoAreaCode']].where(data['sex_code']=='_T').dropna()
countries = data.loc[0:len(data),['geoAreaName']].where(data['sex_code']=='_T').dropna()
uniquecountries=np.unique(countries['geoAreaName'].to_numpy())

# Get means of the countries ICT-skill values
data_ICT=np.zeros(88)
geo_codes = np.zeros(88)
for i in range(88):
    value=np.mean(data_all['latest_value'].where(data_all['geoAreaName']==uniquecountries[i]).dropna())
    code=data_all['geoAreaCode'].where(data_all['geoAreaName']==uniquecountries[i]).dropna().to_numpy()[0]
    data_ICT[i]=value
    geo_codes[i]=code

# Construct the values into pandas dataframe and save it as csv file
aineisto = {'Country': uniquecountries, 'Value': data_ICT, 'geoAreaCode': geo_codes}
data_last = pd.DataFrame(aineisto)
data_last.to_csv("data_ICT.csv")
