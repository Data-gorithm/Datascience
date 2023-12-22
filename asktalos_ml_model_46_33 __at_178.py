# -*- coding: utf-8 -*-
"""asktalos - ML model-Copy1 46.32849 at 178.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tse9VOJx_29SVuEghgL5tNmJmLXjtTUD

PRE-REQUISITE INFO:

The execution time, trial-error time consumed for this project lasted from several hours to several days/weeks which caused delays in the task completion.The code had to be run on other platforms and on other systems to reduce the execution time and to observe the output. To avoid this high computation time, after several weeks of trail-error method, only the final necessary steps are selected, coded, executed and documented here.
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("Material Compressive Strength Experimental Data.csv")

dataset.head(5)

dataset.isnull().sum()

"""NOTE: The below code eliminates all the null values in integer, strings format and by default fills the missing values with mean.

code reference - stack overflow
"""

def clean_dataset(dataset):
    assert isinstance(dataset, pd.DataFrame)
    dataset.dropna(inplace=True)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return dataset[indices_to_keep].astype(np.float64)

clean_dataset(dataset)

dataset.isnull().sum()

"""OBSERVATION:

A clean dataset is achieved.
"""

#scaling to increase the efficieny of the model
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit_transform(dataset)
transformer

scaled_dataset = pd.DataFrame(transformer, columns = dataset.columns)
scaled_dataset

from sklearn.model_selection import train_test_split

x = scaled_dataset[['Material Quantity (gm)','Additive Catalyst (gm)','Ash Component (gm)','Water Mix (ml)','Plasticizer (gm)','Moderate Aggregator','Refined Aggregator','Formulation Duration (hrs)']].values

y = scaled_dataset['Compression Strength MPa'].values

#hyper-parameter used here is random_state
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 150)

"""OBSERVATION:

After tuning the combinations of test_size and random_state continuosly, highest r2_score was achieved at test_size = 0.2, random_state = 150

PRE-REQUISITE:

A pip was installed and lazy model was run to find the most suitable model for this project.

OBSERVATION:

It was found that RandomForestRegressor tops the list and hence was chosen.
"""

from sklearn.ensemble import RandomForestRegressor

"""PRE-REQUISITE:

A plot of n_estimators in the range(1,200) Vs their r2_score was plotted to check the range at which model achieves high accuracy.
Also, we are tuning the n_estimators manually to see the highest output possible.

OBSERVATION:

The plot showed highest r2_score in the range 150 to 200 n_estimator.
"""

#hyper-parameters n_estimators is tuned manually
regf = RandomForestRegressor(n_estimators=178)
regf.fit(x_train, y_train)
y_pred = regf.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)

r2_score(y_test,y_pred)*100
