# insall necessary libraries:
# pip install sklearn pandas tqdm funcsigs pgmpy statsmodels community
# pip install networkx==v1.11
# pip install matplotlib==2.2.3
# link to library info:
# https://pypi.org/project/bnlearn/


import bnlearn as bnlearn
import pandas as pd


# Read the dataframe with the structure to be learned
df = pd.read_csv("X_train_TopLocCust_balanced_allFeatures_pk_non_numeric.csv")
print(df)
model = bnlearn.structure_learning(df)
G = bnlearn.plot(model)