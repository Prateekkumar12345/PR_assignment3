## CODE FOR COMPARATIVE ANALYSIS
# Import necessary libraries for visualization and model training
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K

# Load and preprocess the dataset
file_path = 'database.csv'
data = pd.read_csv(file_path)

# Handle missing values
data = data.dropna(subset=['All Costs', 'Property Damage Costs', 'Lost Commodity Costs', 
                            'Public/Private Property Damage Costs', 'Emergency Response Costs', 
                            'Environmental Remediation Costs', 'Other Costs'])

# Select features and target variable
features = ['Property Damage Costs', 'Lost Commodity Costs', 
            'Public/Private Property Damage Costs', 'Emergency Response Costs', 
            'Environmental Remediation Costs', 'Other Costs']
target = 'All Costs'

# Prepare feature and target datasets
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Outlier Detection - Using Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(X_train))
X_train = X_train[(z_scores < 2.5).all(axis=1)]
y_train = y_train[(z_scores < 2.5).all(axis=1)]

# LSTM Model with Regularization and Early Stopping
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=False))
    model.add(Dropout(0.3))  # Increased Dropout
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Adding Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

lstm_model = create_lstm_model()
history_lstm = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])

# Bayesian Ridge Regression
bayesian_model = BayesianRidge(alpha_1=1e-5, alpha_2=1e-5, lambda_1=1e-5, lambda_2=1e-5)
bayesian_model.fit(X_train, y_train)

# Random Forest Regressor with Hyperparameter Tuning
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=4, random_state=42)
rf_model.fit(X_train, y_train)

# Gradient Boosting Regressor
gboost_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
gboost_model.fit(X_train, y_train)

# XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# SVR - Support Vector Regressor
svr_model = SVR(C=1.0, epsilon=0.2)
svr_model.fit(X_train, y_train)

# Predictions from each model
lstm_predictions = lstm_model.predict(X_test_lstm).flatten()
bayesian_predictions = bayesian_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
gboost_predictions = gboost_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
svr_predictions = svr_model.predict(X_test)

# Stack the predictions for ensemble
stacked_predictions = np.column_stack((bayesian_predictions, rf_predictions, gboost_predictions, xgb_predictions, lstm_predictions, svr_predictions))

# Final Ensemble Model
final_model = BayesianRidge()
final_model.fit(stacked_predictions, y_test)
ensemble_predictions = final_model.predict(stacked_predictions)

# Evaluate the Ensemble Model
from sklearn.metrics import mean_squared_error, r2_score
ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
ensemble_r2 = r2_score(y_test, ensemble_predictions)

print("\nTesting the Ensemble Model:")
print(f'Mean Squared Error: {ensemble_mse}')
print(f'R^2 Score: {ensemble_r2}')

# Create DataFrame for actual vs predicted values
results_df = pd.DataFrame({
    'Actual Value': y_test,
    'Bayesian Ridge Prediction': bayesian_predictions,
    'Random Forest Prediction': rf_predictions,
    'Gradient Boosting Prediction': gboost_predictions,
    'XGBoost Prediction': xgb_predictions,
    'SVR Prediction': svr_predictions,
    'LSTM Prediction': lstm_predictions,
    'Ensemble Prediction': ensemble_predictions
})

# Display the first few rows of the DataFrame
print(results_df.head())

# Visualization: Actual vs Predicted for Ensemble Model
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=ensemble_predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Ensemble Model: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.tight_layout()
plt.show()

# Clear Keras session
K.clear_session()



## CODE FOR PREDICTIVE ANALYSIS
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Load the dataset
file_path = 'database.csv'
data = pd.read_csv(file_path)

# Handle missing values
data = data.dropna(subset=['All Costs', 'Property Damage Costs', 'Lost Commodity Costs', 
                            'Public/Private Property Damage Costs', 'Emergency Response Costs', 
                            'Environmental Remediation Costs', 'Other Costs'])

# Select features and target variable
features = ['Property Damage Costs', 'Lost Commodity Costs', 
            'Public/Private Property Damage Costs', 'Emergency Response Costs', 
            'Environmental Remediation Costs', 'Other Costs']
target = 'All Costs'

# Prepare feature and target datasets
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the Random Forest model
rf_model.fit(X_train_scaled, y_train)

# Make predictions with Random Forest
rf_predictions = rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model performance
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)  # Calculate Mean Absolute Error

# Output the evaluation metrics for Random Forest
print("\nTesting the Random Forest Regression Model:")
print(f'Mean Squared Error: {rf_mse}')
print(f'R^2 Score: {rf_r2}')
print(f'Mean Absolute Error: {rf_mae}')  # Print Mean Absolute Error

# Optionally, display predicted vs actual values for Random Forest
results_rf = pd.DataFrame({'Actual': y_test, 'Random Forest Predicted': rf_predictions})
print(results_rf)

# Gaussian Process Regression
# Define the kernel for the Gaussian Process
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Fit Gaussian Process Regression
gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr_model.fit(X_train_scaled, y_train)

# Make predictions with Gaussian Process
y_pred_gpr, std_gpr = gpr_model.predict(X_test_scaled, return_std=True)

# Evaluate Gaussian Process Model Performance
gpr_mse = mean_squared_error(y_test, y_pred_gpr)
gpr_r2 = r2_score(y_test, y_pred_gpr)
gpr_mae = mean_absolute_error(y_test, y_pred_gpr)  # Calculate Mean Absolute Error

print("\nTesting the Gaussian Process Regression Model:")
print(f'Mean Squared Error: {gpr_mse}')
print(f'R^2 Score: {gpr_r2}')
print(f'Mean Absolute Error: {gpr_mae}')  # Print Mean Absolute Error

# Display predicted vs actual values for Gaussian Process model
results_gpr = pd.DataFrame({'Actual': y_test, 'GPR Predicted': y_pred_gpr})
print(results_gpr)

# Ensemble Predictions
ensemble_predictions = (rf_predictions + y_pred_gpr) / 2

# Evaluate Ensemble Model Performance
ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
ensemble_r2 = r2_score(y_test, ensemble_predictions)
ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)  # Calculate Mean Absolute Error

print("\nTesting the Ensemble Model Performance:")
print(f'Mean Squared Error: {ensemble_mse}')
print(f'R^2 Score: {ensemble_r2}')
print(f'Mean Absolute Error: {ensemble_mae}')  # Print Mean Absolute Error

# Display final predictions vs actual values for Ensemble
results_ensemble = pd.DataFrame({
    'Actual': y_test,
    'Random Forest Predicted': rf_predictions,
    'GPR Predicted': y_pred_gpr,
    'Ensemble Predicted': ensemble_predictions
})
print("\nEnsemble Predictions vs Actual:")
print(results_ensemble)

# Plotting Actual vs Predicted for each model
plt.figure(figsize=(15, 5))

# Random Forest Predictions
plt.subplot(1, 3, 1)
plt.scatter(y_test, rf_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Random Forest Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Gaussian Process Predictions
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_gpr, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Gaussian Process Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Ensemble Predictions
plt.subplot(1, 3, 3)
plt.scatter(y_test, ensemble_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Ensemble Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()
