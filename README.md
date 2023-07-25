# Bike-Sharing-Demand-Prediction-with-Machine-Learning

### Background

Bike sharing systems are new generations of traditional bike rentals where the entire process, from membership, rental, to return, has become automatic. These systems allow users to easily rent a bike from one position and return it to another. With over 500 bike-sharing programs worldwide and more than 500 thousand bicycles, these systems play a crucial role in addressing traffic, environmental, and health issues.

Apart from their real-world applications, bike sharing systems generate data with valuable characteristics, making them attractive for research. Unlike other transport services, bike sharing systems explicitly record the duration of travel and departure and arrival positions, effectively turning them into virtual sensor networks that monitor mobility in the city. This dataset is expected to be useful for detecting significant events and anomalies in the city.

### License and Attribution

If you use this dataset in your own work, you must abide by the licensing terms set forth by the original creators. The use of this dataset in publications requires proper citation to the following publication:

[Hadi Fanaee-T, Joao Gama. "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3](http://dx.doi.org/10.1007/s13748-013-0040-3).

### Contact

For further information about this dataset, please contact Hadi Fanaee-T at hadi.fanaee@fe.up.pt.

### Dataset

The bike-sharing rental process is highly correlated with environmental and seasonal settings. Weather conditions, precipitation, day of the week, season, and hour of the day can all affect rental behaviors. The core data set used in this project is based on the two-year historical log corresponding to years 2011 and 2012 from the Capital Bikeshare system in Washington D.C., USA. The dataset is publicly available [here](http://capitalbikeshare.com/system-data). The data has been aggregated on both an hourly and daily basis, and weather and seasonal information have been extracted and added to it. Weather information is obtained from [Freemeteo](http://www.freemeteo.com).

## Files
	- Readme.txt
	- hour.csv : bike sharing counts aggregated on hourly basis. Records: 17379 hours
	- day.csv - bike sharing counts aggregated on daily basis. Records: 731 days


## Dataset characteristics
### Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
	
	- instant: record index
	- dteday : date
	- season : season (1:springer, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2011, 1:2012)
	- mnth : month ( 1 to 12)
	- hr : hour (0 to 23)
	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	- hum: Normalized humidity. The values are divided to 100 (max)
	- windspeed: Normalized wind speed. The values are divided to 67 (max)
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered

## Notebooks

1. **KNN and Linear Regression:** This notebook explores the bike sharing data using only the 'casual' and 'registered' features and applies K-Nearest Neighbors (KNN) and Linear Regression algorithms for demand prediction.

2. **Random Forest:** In this notebook, we build a Random Forest model to predict bike rental demand based on various environmental and seasonal settings.

## Prerequisites
To run the notbooks , make sure you have the following dependencies installed:

Python 3.x
Sklearn
NumPy
Pandas
Matplotlib
Seaborn

## How to Use

1. Clone the repository to your local machine.
2. Set up the virtual environment and install the dependencies as explained in the 'Prerequisites' section.
3. Open the notebooks using Jupyter or any compatible environment.
4. Run the notebooks cell by cell to explore the code and results.

______________________________________________________________________________________________________________________________________________________________

## Notebook 1: KNN and Linear Regression

In this notebook, we explore the relationship between the bike rental count ('cnt') and two features: 'casual' and 'registered'. The main steps of the notebook are as follows:

1. **Data Visualization**: We start by visualizing the dataset to understand the relationship between the 'temp' and 'atemp' features and the bike rental count ('cnt'). We use scatter plots and a heatmap to analyze correlations.

2. **Data Preparation**: We create a subset of the dataset containing only the 'casual' and 'registered' features as inputs ('x') and the total rental bike counts ('cnt') as the target ('y').

3. **Train-Test Split**: We split the data into 80% training and 20% testing sets for model evaluation.

4. **Data Scaling**: To ensure fair comparison and accurate results, we scale the features using StandardScaler.

5. **KNN Regression**: We use K-Nearest Neighbors (KNN) regression to predict bike rental counts based on the 'casual' and 'registered' features. We perform hyperparameter tuning using GridSearchCV.

6. **Linear Regression**: We also apply Linear Regression to predict bike rental counts with the same features.

7. **Model Evaluation**: Finally, we evaluate both models using the R-squared metric to assess their performance.

8.   Model Evaluation Results

The R-squared scores for the KNN and Linear Regression models are as follows:

- KNN Regression: R-squared = 0.998107
- Linear Regression: R-squared = 1.000000


## Notebook 2: Random Forest

In this notebook, we employ the Random Forest algorithm to predict bike rental demand based on various environmental and seasonal settings. The key steps in this notebook are as follows:

1. **Data Preparation**: We create a subset of the dataset containing relevant features as inputs ('x') and the total rental bike counts ('cnt') as the target ('y'). The features used include 'season', 'temp', 'atemp', 'hum', 'windspeed', 'casual', and 'registered'.

2. **Train-Test Split**: Similar to the previous notebooks, we split the data into 80% training and 20% testing sets for model evaluation.

3. **Hyperparameter Tuning**: We perform hyperparameter tuning using GridSearchCV to find the best combination of hyperparameters for the Random Forest model. The tuned hyperparameters include 'n_estimators', 'max_depth', and 'min_samples_leaf'.

4. **Feature Importance**: We analyze the feature importance provided by the trained Random Forest model to identify the most significant features affecting bike rental demand.

5. **Model Evaluation**: We evaluate the Random Forest model using the R-squared metric, which measures the goodness of fit. Additionally, we calculate the Mean Absolute Error (MAE) to assess the model's prediction accuracy.

6. Model Evaluation Results

The Random Forest model achieved the following evaluation metrics on the test set:

- R-squared (r2): 0.997350
- Mean Absolute Error (MAE): 66.787267

The Random Forest model demonstrates strong predictive performance and has identified 'atemp' (90%) and 'temp' (10%) as the top two significant features affecting bike rental demand.

