# Import
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np

dataset = np.loadtxt("D:/00_SFU/00_Graduate_Courses/00_CMPT741_DataMining/Project/2019-741_Data/training_data_preprocessed.csv", delimiter=",")
y = dataset[:,:1]
qid = dataset[:,1]
X = dataset[:,2:]
print("\nDataset Dimensions : ",dataset.shape)

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"alpha": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.2,1.5]}

lasso = Lasso(normalize=True)

# Instantiate the RandomizedSearchCV object: lasso_cv
lasso_cv = RandomizedSearchCV(lasso, param_dist, cv=5)

# Fit it to the data
lasso_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Parameters: {}".format(lasso_cv.best_params_))
print("Best score is {}".format(lasso_cv.best_score_))


# LASSO SETUP
lasso = Lasso (alpha = lasso_cv.best_params_['alpha'],normalize=True)
lasso_coef = lasso.fit(X,y).coef_
lasso_coef_positive = lasso_coef[lasso_coef > 0]
plt.plot(range(len(lasso_coef_positive)),lasso_coef_positive)
plt.xticks(range(len(lasso_coef_positive)),range(0,58),rotation=60)
plt.ylabel('coefficients')
plt.show()

features_selected = np.where(np.array(lasso_coef) > 0)[0]
print("Features Selected [%d]:" %len(features_selected),features_selected)
X = X[:,features_selected]
print(X.shape)