import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
import time
import os
import shap

import tensorflow as tf    
tf.compat.v1.disable_v2_behavior() # <-- HERE !
TF_ENABLE_ONEDNN_OPTS=0

start_time = time.time()

# Load data from the CSV file
data = pd.read_csv('combine_data_climat_next_02.csv')

# Reverse the data along the rows
data = np.flip(data.values, axis=0)

# Extract features and target
features = data[:, 1:]  # All columns except the first (time)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Define the number of data points for training, validation, and test
train_size = 2880 * 6
val_size = 1440 * 6
test_size = len(features) - train_size - val_size

# Split the data into training, validation, and test sets
train_data = scaled_features[:train_size, :]
val_data = scaled_features[train_size:train_size + val_size, :]
test_data = scaled_features[train_size + val_size:, :]

# Define a function to create sequences for the LSTM model
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :-1])  # Use all columns as features except the last one
        y.append(data[i + time_steps, -1])  # Predicting the last column
    return np.array(X), np.array(y)

# Define the number of time steps for the LSTM model
time_steps = 24 * 6  # You may adjust this based on your data and problem

# Create sequences for training, validation, and test sets
X_train, y_train = create_sequences(train_data, time_steps)
X_val, y_val = create_sequences(val_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# Define the BiLSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=96, activation='relu'), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

""" ---------- Make predictions on the test set ---------
"""
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to their original scale
predictions = scaler.inverse_transform(np.hstack((X_test[:, -1], predictions.reshape(-1, 1))))
actual_values = scaler.inverse_transform(np.hstack((X_test[:, -1], y_test.reshape(-1, 1))))

# Reverse the predictions and actual values along the rows
predictions = np.flip(predictions, axis=0)
actual_values = np.flip(actual_values, axis=0)

# Extract timestamps corresponding to the test set
test_timestamps = np.flip(data[train_size + val_size + time_steps:, 0])

# Convert predictions, actual values, and timestamps arrays to DataFrame
df_predictions = pd.DataFrame({
    'Timestamp': test_timestamps,
    'Actual Values': actual_values[:, -1],
    'Predicted Values': predictions[:, -1]
})

# Save DataFrame to CSV file
main_file_name = os.path.splitext(__file__)[0]
main_file_name_csv = main_file_name + ".csv"
df_predictions.to_csv(main_file_name_csv, index=False)

# Calculate RMSE for the predictions
# Identify NaN values in both arrays
nan_indices = np.isnan(actual_values[:, -1]) | np.isnan(predictions[:, -1])
actual_values_filtered = actual_values[:, -1][~nan_indices]
predictions_filtered = predictions[:, -1][~nan_indices]

rmse = np.sqrt(mean_squared_error(actual_values_filtered, predictions_filtered))
r2 = r2_score(actual_values_filtered, predictions_filtered)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_values[:, -1], label='Actual Values')
plt.plot(predictions[:, -1], label='Predicted Values')
plt.title('Actual vs. Predicted Values for value1')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
plt.show()

# Measure the end time
end_time = time.time()
total_running_time = end_time - start_time
print(f"Total running time: {total_running_time} seconds")

# Write results to a file
main_file_name_txt = main_file_name + ".txt"
with open(main_file_name_txt, 'w') as f:
    f.write("Total running time: {} seconds\n".format(total_running_time))
    f.write("R2 on the test set: {}\n".format(r2))
    f.write("RMSE on the test set: {}\n".format(rmse))


""" ------------ Calculate SHAP values ------------
"""

length = 144*14

background = X_train[np.random.choice(X_train.shape[0], 144*3, replace=False)]
explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = explainer.shap_values(X_test[:length])

# Define feature names
feature_names = ['Balise A','Fatouville','Pressure','Fx','Fy','Precipitation','Temperature','Telemac']

# Plot SHAP values for each feature
fig = plt.figure()
# Reshape the shap_values for visualization
shap_values_reshaped = np.concatenate(shap_values, axis=1).reshape(-1, X_test.shape[2])

# Plot SHAP summary plot
shap.summary_plot(shap_values_reshaped, X_test[:length].reshape(-1, X_test.shape[2]), feature_names=feature_names)


main_file_name_png = main_file_name + ".png"
fig.savefig(main_file_name_png, format='png', dpi=300)

## waterfall plot
plt.figure(figsize=(10, 6))
# Select a specific time step for waterfall plot (e.g., first time step)
time_step_idx = 0

# Generate waterfall plot for the selected time step
sample_idx = 0  # Choose a specific sample to visualize

# Flatten the SHAP values for the specific time step
shap_values_flat = shap_values[sample_idx][time_step_idx].flatten()

# Generate waterfall plot for the selected time step
shap.waterfall_plot(shap.Explanation(values=shap_values_flat, base_values=explainer.expected_value, data=X_test[:length][sample_idx], feature_names=feature_names), max_display=6)

""" ------ Save data ---------
"""
import gzip
import pickle

# Define the file path
file_path = main_file_name + ".pkl.gz"

# Create a dictionary to store the variables
variables_to_save = {
    'shap_values': shap_values,
    'X_test': X_test,
    'feature_names': feature_names,
    'length': length
}

# Save the variables to a file using pickle
with gzip.open(file_path, 'wb') as file:
    pickle.dump(variables_to_save, file)

print("Variables saved successfully.")
