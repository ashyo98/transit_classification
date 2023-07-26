# transit_classification
Exploration of machine learning methods of detecting transits in Kepler data.

## Summary
Never before has the detection and characterization of exoplanets via transit photometry been as promis
ing and feasible as it is now, due to the increasing breadth and sensitivity of time domain optical surveys.
Visually identifying transits in stellar lightcurves (flux as a function of time) is impractically time-consuming
and tedious, but machine learning is uniquely suited to the task of identifying which lightcurves contain
transits. In this project We explore and evaluate several supervised machine learning algorithms to classify
lightcurves by whether or not they are likely to contain exoplanet transits.

## Data and pre-processing
`utils.py` contains functions to download data and construct light curve datasets using the **LightKurve** Python package, with optional additional pre-processing and feature engineering. `download_data.py` provides and example call to our data download function.

## Traditional ML algorithms
We explore the use of KNN, Random Forest, and Logistic Regression classifiers in `Standard_Algorithms.ipynb`.
We explore SVMs in `SVM.py`.

## Ensemble methods
We explore AdaBoost and Gradient Boost classifiers in `Ensembles.ipynb`.

## Deep learning 
We explore various nueral netwrok architectures in `Transits_NN_Notebook.ipynb`.
Constructed models are in `utils.py`.
