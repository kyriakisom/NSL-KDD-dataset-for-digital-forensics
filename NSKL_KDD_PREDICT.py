# Module Imports
import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import itertools
import random

print('Welcome!')

# Fetch the training file
file_path_20_percent = 'KDDTrain+_20Percent.txt'
file_path_full_training_set = 'KDDTrain+.txt'
file_path_test = 'KDDTest+.txt' 

df = pd.read_csv(file_path_full_training_set)
test_df = pd.read_csv(file_path_test)

# Add the column labels
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate', 'attack', 'level'
]

df.columns = columns
test_df.columns = columns

# Map normal to 0, all attacks to 1
is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)

df['attack_flag'] = is_attack
test_df['attack_flag'] = test_attack

# Define attack types
dos_attacks = [
    'apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 
    'smurf', 'teardrop', 'udpstorm', 'worm'
]
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
access_attacks = [
    'ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 'multihop', 'named', 'phf', 
    'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 
    'xclock', 'xsnoop'
]

attack_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']

# Helper function to map attack types
def map_attack(attack):
    if attack in dos_attacks:
        attack_type = 1
    elif attack in probe_attacks:
        attack_type = 2
    elif attack in privilege_attacks:
        attack_type = 3
    elif attack in access_attacks:
        attack_type = 4
    else:
        attack_type = 0
    return attack_type

attack_map = df.attack.apply(map_attack)
df['attack_map'] = attack_map

test_attack_map = test_df.attack.apply(map_attack)
test_df['attack_map'] = test_attack_map

# Encode categorical features
features_to_encode = ['protocol_type', 'service', 'flag']
encoded = pd.get_dummies(df[features_to_encode])
test_encoded_base = pd.get_dummies(test_df[features_to_encode])

# Handle missing columns in test set
test_index = np.arange(len(test_df.index))
column_diffs = list(set(encoded.columns.values) - set(test_encoded_base.columns.values))
diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

# Reorder columns
column_order = encoded.columns.to_list()
test_encoded_temp = test_encoded_base.join(diff_df)
test_final = test_encoded_temp[column_order].fillna(0)

# Get numeric features
numeric_features = ['duration', 'src_bytes', 'dst_bytes']

# Prepare training and test sets
to_fit = encoded.join(df[numeric_features])
test_set = test_final.join(test_df[numeric_features])

# Create target classifications
binary_y = df['attack_flag']
multi_y = df['attack_map']
test_binary_y = test_df['attack_flag']
test_multi_y = test_df['attack_map']

# Split the data
binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(to_fit, binary_y, test_size=0.6)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(to_fit, multi_y, test_size=0.6)

# Convert to DataFrames
binary_train_df = pd.DataFrame(binary_train_X)
binary_val_df = pd.DataFrame(binary_val_X)
binary_train_y_df = pd.DataFrame(binary_train_y)
binary_val_y_df = pd.DataFrame(binary_val_y)
multi_train_df = pd.DataFrame(multi_train_X)
multi_val_df = pd.DataFrame(multi_val_X)
multi_train_y_df = pd.DataFrame(multi_train_y)
multi_val_y_df = pd.DataFrame(multi_val_y)

# Save to CSV
binary_train_df.to_csv('binary_train_X.csv', index=False)
binary_val_df.to_csv('binary_val_X.csv', index=False)
binary_train_y_df.to_csv('binary_train_y.csv', index=False)
binary_val_y_df.to_csv('binary_val_y.csv', index=False)
multi_train_df.to_csv('multi_train_X.csv', index=False)
multi_val_df.to_csv('multi_val_X.csv', index=False)
multi_train_y_df.to_csv('multi_train_y.csv', index=False)
multi_val_y_df.to_csv('multi_val_y.csv', index=False)

print(binary_train_X.info())
print(binary_train_X.sample(5))

# Model for binary classification
binary_model = RandomForestClassifier()
binary_model.fit(binary_train_X, binary_train_y)
binary_predictions = binary_model.predict(binary_val_X)

# Calculate and display base accuracy
base_rf_score = accuracy_score(binary_predictions, binary_val_y)
print(base_rf_score)

# Define the list of models to test
models = [
    RandomForestClassifier(),
    LogisticRegression(max_iter=250),
    KNeighborsClassifier(),
]

# Capture the performance of each model
model_comps = []

# Evaluate each model
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, binary_train_X, binary_train_y, scoring='accuracy')
    for count, accuracy in enumerate(accuracies):
        model_comps.append((model_name, count, accuracy))

# Plot model performance
result_df = pd.DataFrame(model_comps, columns=['model_name', 'count', 'accuracy'])
pivot_df = result_df.pivot(index='count', columns='model_name', values='accuracy')

plt.figure(figsize=(10, 6))
boxplot = pivot_df.boxplot()
plt.xticks(rotation=45)
plt.title('Model Performance Comparison')
plt.xlabel('Model Name')
plt.ylabel('Accuracy')
plt.show()

# Model for multi-class classification
multi_model = RandomForestClassifier()
multi_model.fit(multi_train_X, multi_train_y)
multi_predictions = multi_model.predict(multi_val_X)

# Get the score
print(accuracy_score(multi_predictions, multi_val_y))

print('-------------------------------------------------------------------------')
print('Model ANN:')

# Convert boolean to integer in the target variables
binary_train_y = binary_train_y.astype(int)
binary_val_y = binary_val_y.astype(int)

# Convert any boolean to integer in the features if needed
binary_train_X = binary_train_X.astype(float)
binary_val_X = binary_val_X.astype(float)

# Define the ANN model
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_shape=(binary_train_X.shape[1],)))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN model on the Training set
ann.fit(binary_train_X, binary_train_y, batch_size=32, epochs=100, validation_split=0.2)

# Predicting the results of the Validation set
ann_predictions = (ann.predict(binary_val_X) > 0.5).astype("int32")

# Evaluating the model's performance
ann_accuracy = accuracy_score(binary_val_y, ann_predictions)
print(f"ANN Model Accuracy: {ann_accuracy}")

# Evaluating the model on the validation set
val_loss, val_accuracy = ann.evaluate(binary_val_X, binary_val_y)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Forensic process
def forensic_process(predictions, threshold):
    for i, prediction in enumerate(predictions):
        distinctiveness = abs(prediction - 0.5)
        if distinctiveness < threshold:
            print(f"Attack detected at index {i}, distinctiveness: {distinctiveness}")
            response = compute_response(distinctiveness)
            print(f"Response: {response}")
            distinctiveness = response
    print("Forensic process completed")

def compute_response(distinctiveness):
    return distinctiveness * 2

# Getting the predictions
predictions = ann.predict(binary_train_X)

# Apply the forensic process with a threshold for attack detection
forensic_process(predictions, threshold=0.2)
