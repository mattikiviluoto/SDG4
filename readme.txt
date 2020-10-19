The Global Goal #4: QUALITY EDUCATION
A toolkit for researchers and decisionmakers around the globe

Quality Education  a Catalyst for a Better World
- There is probably no greater way to improve the living conditions and well-being of people in any country than quality education.
- Finland is a good global example of a poor developing nation which through quality education rose to its current global position
- That’s why it’s crucially important to provide tools for researchers and policy-makers to help with making the most impactful decisions to promote Quality Education

For Whom?
For researchers with only basic Python skills 
- use the prepared data and ready-made models to conduct your research
For Data Scientists 
- use our work as a starting point, apply our models for new data and develop the models further

The Package
- Full human readable listing of all SDG4 indicators available from the UN SDG API
- Preprocessed CIA factbook background data from countries across the globe
- A model to use a country’s CIA data to predict its missing values in the UN SDG data.
- The user can freely choose both the explanatory variables from the CIA data and the SDG data to use as the target of either Linear Regression or Random Forest Regression.
- Tools for exploratory data analysis are provided, including Principal Component Analysis and k-Means Clustering with user defined number of dimensions and clusters
- Further options to narrow down the list of countries to analyze
- We have used cross-validation to get an idea how accurately the predictive model will perform in practice, typically the Random Forest Regressor outperforms Linear Regression.
- Scatterplots of the real and predicted SDG values as y with each of the chosen explanatory variable as x for the chosen variables
- Clear visualizations make it easy to detect outlier data points. These outliers can be excluded when building the regression model by user's discretion.
- The package will be made available for everyone to use and to contribute on GitHub

Known Issues
- In the UN SDG data there is often a significant number of countries with no data available for the different indicators
- There are some countries that due to political reasons are not statistically treated in the same way in the CIA factbook and the UN SDG data – most importantly Taiwan and Palestine
- We did not have resources to fine tune the design of the visualizations at this point as they are made mainly for research purposes


