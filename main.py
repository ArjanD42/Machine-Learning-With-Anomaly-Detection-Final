import inline
import matplotlib
import numpy as np
import pandas as pd

random_seed = np.random.RandomState(12)

# Creating a set of normal observations to be used as training data
X_train = 0.5 * random_seed.randn(500, 2)
X_train = np.r_[X_train + 3, X_train]
X_train = pd.DataFrame(X_train, columns=["x", "y"])

# Generating A testing set also containing normal observations
X_test = 0.5 * random_seed.randn(500, 2)
X_test = np.r_[X_test + 3, X_test]
X_test = pd.DataFrame(X_test, columns=["x", "y"])

# Generating A set of outliers to the main distribution
X_outliers = random_seed.uniform(low=-5, high=5, size=(50, 2))
X_outliers = pd.DataFrame(X_outliers, columns=["x", "y"])

# Outputting The Generated Data
# %matplotlib inline -- Needed if using Jupyter
import matplotlib.pyplot as plt

p1 = plt.scatter(X_train.x, X_train.y, c="white", s=50, edgecolors="black")
p2 = plt.scatter(X_train.x, X_train.y, c="green", s=50, edgecolors="black")
p3 = plt.scatter(X_outliers.x, X_outliers.y, c="blue", s=50, edgecolors="black")
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.legend(
    [p1, p2, p3],
    ["training set", "normal testing set", "anomalous testing set"],
    loc="lower right",
)

plt.show()

# Training The Isolation Forest Model On Training Data

from sklearn.ensemble import IsolationForest

clf = IsolationForest()
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# Appending The Labels to X_outliers
X_outliers = X_outliers.assign(pred=y_pred_outliers)
print(X_outliers.head())

# Plotting Isolation Forests Predictions On The Outliers For Accuracy Testing

p1 = plt.scatter(X_train.x, X_train.y, c="white", s=50, edgecolors="black")
p2 = plt.scatter(
    X_outliers.loc[X_outliers.pred == -1, ["x"]],
    X_outliers.loc[X_outliers.pred == -1, ["y"]],
    c="blue",
    s=50,
    edgecolors="black",
)
p3 = plt.scatter(
    X_outliers.loc[X_outliers.pred == 1, ["x"]],
    X_outliers.loc[X_outliers.pred == 1, ["y"]],
    c="red",
    s=50,
    edgecolors="black",
)

plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.legend(
    [p1, p2, p3],
    ["training observations", "detected outliers", "incorrectly labeled outliers"],
    loc="lower right",
)

plt.show()

# Append predicted labels to X_test
X_test = X_test.assign(pred=y_pred_test)
print(X_test.head())
