

# %%

#Loading the necessary libraries for creating a NN with Keral

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


# %%
#loading the dataset
IMU_data = pd.read_csv(r"C:\Users\Ronald Chitauro\Pictures\Deep Learning Practice\Dataset\WISDM_ar_v1.1_raw.csv")
IMU_data.set_index('Activity')
IMU_data.head(5)
