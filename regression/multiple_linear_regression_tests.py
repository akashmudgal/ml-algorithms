import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiple_linear_regression import MultipleLinearRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# read the csv file
df = pd.read_csv(r"D:\Downloads\Real estate.csv")
df = df.drop(columns=['X1 transaction date']).to_numpy()

non_scaled_X_train = df[:,0:-1]
non_scaled_y_train = df[:,-1]


scaler = StandardScaler()
df = scaler.fit_transform(df)

# separate X_train, Y_train from the pandas dataframe
X_train = df[:,0:-1]
y_train = df[:,-1]

regressor = MultipleLinearRegressor(lr=0.6,n_iters=1500)
regressor.fit(X_train,y_train)
y_predicted = regressor.predict(X_train)
cost = regressor.compute_cost(X_train,y_train)

sk_regressor = LinearRegression()
sk_regressor.fit(X_train,y_train)
y_predicted_sk = sk_regressor.predict(X_train)
sk_cost = mean_squared_error(y_train,y_predicted_sk)

plt.figure(figsize=(8, 6))
plt.scatter(y_train,y_predicted_sk, color='blue', label='Predictions vs Actual')
plt.plot([np.min(y_predicted_sk), np.max(y_predicted_sk)], [np.min(y_predicted_sk), np.max(y_predicted_sk)], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter Plot: Predictions vs Actual')
plt.legend()
plt.grid(True)
plt.show()
print("Cost of our model: ",cost)
print("Cost of sklearn model: ",sk_cost)
