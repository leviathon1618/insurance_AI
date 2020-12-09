import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('insurance_dataset.csv', encoding='ISO-8859-1')
print(data)

#sns.pairplot(data)
#plt.show()


inputs = data.drop(['gender', 'smoker','charges'], axis = 1)
#Show Input Data
print(inputs)

#Show Input Shape
#print("Input data Shape=",inputs.shape)

#Create output dataset from data
output = data['charges']
#Show Output Data
#print(output)

#Transform Output
output = output.values.reshape(-1, 1)
#Show Output Transformed Shape
#print("Output Data Shape=",output.shape)

#Scale input
scaler_in = MinMaxScaler()
input_scaled = scaler_in.fit_transform(inputs)
#print(input_scaled)


#Scale output
scaler_out = MinMaxScaler()
output_scaled = scaler_out.fit_transform(output)
#print(output_scaled)



#Create model
model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
#print(model.summary())


#Train model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
epochs_hist = model.fit(input_scaled, output_scaled, epochs=80, batch_size=30, verbose=1, validation_split=0.2)
print(epochs_hist.history.keys()) #print dictionary keys


#Plot the training graph to see how quickly the model learns
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
#plt.show()


# Evaluate model
# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 
# ***(Note that input data must be normalized)***
# dataset for testing:
# 18,	31.92,	0,	0,	0-
# 31,	25.94,	1,	0,	1-
# 31,	31.07,	3,	0,	1-
# 42,	32.87,	0,	0,	0-
# 39,	26.32,	2,	0,	0-
# 51,	30.03,	1,	0,	1-
# 52,	38.6,	2,	0,	1-
# 50,	30.97,	3,	0,	1-
# 23,	33.4,	0,	0,	0-
# 52,	44.7,	3,	0,	0-
# 57,	25.74,	2,	0,	0-
# 62,	38.83,	0,	0,	1-
# 61,	33.54,	0,	0,	1-
# 35,	39.71,	4,	0,	1-
# 62,	26.7,	0,	1,	1-
# 61,	29.07,	0,	1,	0-
# 42,	40.37,	2,	1,	0-


input_test_sample = np.array([[42,	40.37,	2,	1,	0]])
#input_test_sample2 = np.array([[1, 46.73, 61370.67, 9391.34, 462946.49]])
	

#Scale input test sample data
input_test_sample_scaled = scaler_in.transform(input_test_sample)

#Predict output
output_predict_sample_scaled = model.predict(input_test_sample_scaled)

#Print predicted output
print('Predicted Output (Scaled) =', output_predict_sample_scaled)

#Unscale output
output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
print('Predicted Output / Purchase Amount ', output_predict_sample)