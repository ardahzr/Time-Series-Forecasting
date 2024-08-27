import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

data = pd.read_csv('veri_SEQDAYS4.csv')
data["SPO2/O2"] = data["SPO2"] / data["O2_CONCENTRATION"]
data["PO2/O2"] = data["PO2"] / data["O2_CONCENTRATION"]

data['OLCUMZAMANI'] = pd.to_datetime(data['OLCUMZAMANI'])
data.set_index('OLCUMZAMANI', inplace=True)


time_steps = 7  # Increase time steps for more context
n_future = 1

X, y = [], []

for hastano, group in data.groupby('HASTANO'):
    group = group.sort_index()
    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(group)

    if len(group) > time_steps:
        # Target PO2 column
        X_seq, y_seq = create_sequences(data_scaled, data_scaled[:, group.columns.get_loc('PO2')], time_steps)

        X.append(X_seq)
        y.append(y_seq)

# Ensure that X and y have compatible shapes
if X and y:
    X = np.concatenate(X)
    y = np.concatenate(y)

    # Reshape y to be 2D for compatibility
    y = y.reshape(-1, 1)

    # Build the Model
    model = Sequential()
    model.add(GRU(100, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))  # prevent overfitting
    model.add(LSTM(50, activation='tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(n_future))

    model.compile(optimizer="nadam", loss='mae')

    # Train the Model
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    history = model.fit(X, y, epochs=200, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

    # Prediction and Results
    y_true = []
    y_pred = []

    for hastano, group in data.groupby('HASTANO'):
        group = group.sort_index()
        scaler = MinMaxScaler()

        data_scaled = scaler.fit_transform(group)

        if len(group) >= time_steps:  # Only predict if there are enough data points
            last_day_data = data_scaled[-time_steps:].reshape(1, time_steps, data_scaled.shape[1])

            prediction = model.predict(last_day_data)
            PO2_prediction_scaled = np.zeros((1, data_scaled.shape[1]))
            PO2_prediction_scaled[:, group.columns.get_loc('PO2')] = prediction
            PO2_prediction = scaler.inverse_transform(PO2_prediction_scaled)[:, group.columns.get_loc('PO2')]

            y_true.append(group['PO2'].values[-1])
            y_pred.append(PO2_prediction[0])

            print(f"Patiend id: {hastano}")
            print(f"Real PO2: {group['PO2'].values[-1]}")
            print(f"Predicted PO2: {PO2_prediction[0]}")
            print("\n")

    # Calculate Metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"R^2 Score: {r2}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
else:
    print("Not enough data decrease time_steps.")
