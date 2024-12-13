import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt


# Load the CSV files
binary_train_X = pd.read_csv('binary_train_X.csv')
binary_val_X = pd.read_csv('binary_val_X.csv')
binary_train_y = pd.read_csv('binary_train_y.csv')
binary_val_y = pd.read_csv('binary_val_y.csv')

# Ensure binary labels are integers
binary_train_y = binary_train_y.astype(int)
binary_val_y = binary_val_y.astype(int)

# Convert any boolean to integer in the features as well if needed
binary_train_X = binary_train_X.astype(float)
binary_val_X = binary_val_X.astype(float)

# Scaling features
sc_X = StandardScaler()
binary_train_X_scaled = sc_X.fit_transform(binary_train_X)
binary_val_X_scaled = sc_X.transform(binary_val_X)

# Reshaping features to be 3D [samples, timesteps, features] for LSTM
# Assuming 1 timestep for simplicity, can be adjusted as needed
binary_train_X_scaled = binary_train_X_scaled.reshape((binary_train_X_scaled.shape[0], 1, binary_train_X_scaled.shape[1]))
binary_val_X_scaled = binary_val_X_scaled.reshape((binary_val_X_scaled.shape[0], 1, binary_val_X_scaled.shape[1]))

# Building the LSTM model
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(binary_train_X_scaled.shape[1], binary_train_X_scaled.shape[2])))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the RNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the RNN to the Training set
history = model.fit(binary_train_X_scaled, binary_train_y, epochs=100, batch_size=32, validation_data=(binary_val_X_scaled, binary_val_y))

# Plotting the accuracy and loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Evaluating the model on the validation set
val_loss, val_accuracy = model.evaluate(binary_val_X_scaled, binary_val_y)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")




# Evaluating the model on the validation set
val_loss, val_accuracy = model.evaluate(binary_val_X_scaled, binary_val_y)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Forensic process
# Step 1: For Forensics Process Do
def forensic_process(predictions, threshold):
    for i, prediction in enumerate(predictions):
        # Step 2: Check the Distinctiveness <-Distinctiveness [Forensics Process]
        distinctiveness = abs(prediction - 0.5)  # assuming 0.5 as the decision boundary for distinctiveness
        
        # Step 3: A <-- attack detection [X]
        if distinctiveness < threshold:
            print(f"Attack detected at index {i}, distinctiveness: {distinctiveness}")
            
            # Step 4: if (Compute the process response [Distinctiveness]) then
            response = compute_response(distinctiveness)
            print(f"Response: {response}")
            
            # Step 5: X. Compute the process [Distinctiveness] = Compute the process response [Distinctiveness], Forensics Process
            distinctiveness = response
        # Step 6: End if Condition
        # Step 7: Terminating the Condition of for.
    print("Forensic process completed")

def compute_response(distinctiveness):
    # Dummy response computation based on distinctiveness
    return distinctiveness * 2

# Getting the predictions
predictions = model.predict(binary_val_X_scaled)

# Apply the forensic process with a threshold for attack detection
forensic_process(predictions, threshold=0.2)