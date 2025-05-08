### Author: Bill Zhu 
### Date: 05/05/2025
### Description: This script is used to predict the claim labels and retrieve evidences for a given dataset using a pre-trained model.

import numpy as np

loaded_array = np.load('train_vectors.npy')

print(loaded_array.shape)  # Check the shape of the loaded array
print(loaded_array)  # Print the loaded array to verify its contents