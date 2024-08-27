import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD


def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


#Load and Prepare Data
data = pd.read_csv('veri_SEQDAYS4.csv') #Load your own data and prepare
data["SPO2/O2"] = data["SPO2"] / data["O2_CONCENTRATION"]
data["PO2/O2"] = data["PO2"] / data["O2_CONCENTRATION"]

data['OLCUMZAMANI'] = pd.to_datetime(data['OLCUMZAMANI'])
data.set_index('OLCUMZAMANI', inplace=True)

time_steps = 7
n_future = 1

# Define a list of optimizers, loss functions, and activations to try
optimizers = [Adam(), RMSprop(), Nadam(), SGD(learning_rate=0.01, momentum=0.9)]
loss_functions = ['mae', 'mse', 'huber']
activations = ['relu', 'tanh', 'sigmoid']

best_r2 = -1  #best R^2 score
best_params = {}  #store the best parameters

for optimizer_class in [Adam, RMSprop, Nadam, SGD]:
    for loss_function in loss_functions:
        for activation in activations:
            for num_layers in range(1, 4):  # Try 1 to 3 LSTM layers
                for units in [50, 100, 150]:  # Try different number of units
                    for dropout_rate in [0.1, 0.2, 0.3]:  # Try different dropout rates
                        optimizer = optimizer_class()  # Create a new optimizer instance
                        print(f"Optimizer: {optimizer.__class__.__name__}, Loss: {loss_function}, "
                              f"Activation: {activation}, Layers: {num_layers}, Units: {units}, "
                              f"Dropout: {dropout_rate}")
                        X, y = [], []

                        for hastano, group in data.groupby('HASTANO'):
                            group = group.sort_index()
                            scaler = MinMaxScaler()

                            data_scaled = scaler.fit_transform(group)

                            if len(group) > time_steps:
                                X_seq, y_seq = create_sequences(data_scaled,
                                                                  data_scaled[:, group.columns.get_loc('PO2')],
                                                                  time_steps)

                                X.append(X_seq)
                                y.append(y_seq)

                        if X and y:
                            X = np.concatenate(X)
                            y = np.concatenate(y)
                            y = y.reshape(-1, 1)

                       
                            model = Sequential()

                            model.add(Bidirectional(LSTM(units, activation=activation, return_sequences=True),
                                                    input_shape=(X.shape[1], X.shape[2])))
                            model.add(Dropout(dropout_rate))

                            # Add more LSTM layers if num_layers > 1
                            for _ in range(num_layers - 1):
                                model.add(Bidirectional(LSTM(units, activation=activation, return_sequences=True)))
                                model.add(Dropout(dropout_rate))

                            model.add(Bidirectional(LSTM(units, activation=activation)))
                            model.add(Dropout(dropout_rate))
                            model.add(Dense(n_future))

                            model.compile(optimizer=optimizer, loss=loss_function)

                            #Train the Model
                            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            history = model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2,
                                                callbacks=[early_stopping], verbose=0)

                            #Prediction and Results
                            y_true = []
                            y_pred = []

                            for hastano, group in data.groupby('HASTANO'):
                                group = group.sort_index()
                                scaler = MinMaxScaler()

                                data_scaled = scaler.fit_transform(group)

                                if len(group) >= time_steps:
                                    last_day_data = data_scaled[-time_steps:].reshape(1, time_steps,
                                                                                    data_scaled.shape[1])

                                    prediction = model.predict(last_day_data)
                                    PO2_prediction_scaled = np.zeros((1, data_scaled.shape[1]))
                                    PO2_prediction_scaled[:, group.columns.get_loc('PO2')] = prediction
                                    PO2_prediction = scaler.inverse_transform(PO2_prediction_scaled)[:,
                                                    group.columns.get_loc('PO2')]

                                    y_true.append(group['PO2'].values[-1])
                                    y_pred.append(PO2_prediction[0])

                            # Calculate Metrics
                            r2 = r2_score(y_true, y_pred)
                            mse = mean_squared_error(y_true, y_pred)
                            mae = mean_absolute_error(y_true, y_pred)

                            print(f"R^2 Score: {r2}")
                            print(f"Mean Squared Error (MSE): {mse}")
                            print(f"Mean Absolute Error (MAE): {mae}")

                            # Update best parameters if current R^2 is better
                            if r2 > best_r2:
                                best_r2 = r2
                                best_params = {
                                    'optimizer': optimizer.__class__.__name__,
                                    'loss_function': loss_function,
                                    'activation': activation,
                                    'num_layers': num_layers,
                                    'units': units,
                                    'dropout_rate': dropout_rate,
                                    'r2': best_r2,
                                    'mse': mse,
                                    'mae': mae
                                }

print("\nBest parameters: ")
print(best_params)
