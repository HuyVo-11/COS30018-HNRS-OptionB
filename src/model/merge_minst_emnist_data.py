import numpy as np
import pandas as pd
from keras.datasets import mnist

def save_merged_data(emnist_train_path, emnist_test_path):
    # --- 1. Load MNIST (Keras) ---
    print("Loading MNIST...")
    (x_train_m, y_train_m), (x_test_m, y_test_m) = mnist.load_data()

    # --- 2. Load EMNIST (CSV) ---
    print("Loading EMNIST from CSV...")
    train_emnist = pd.read_csv(emnist_train_path, header=None)
    test_emnist = pd.read_csv(emnist_test_path, header=None)

    # Process Train EMNIST
    y_train_e = train_emnist.iloc[:, 0].values
    x_train_e = train_emnist.iloc[:, 1:].values.reshape(-1, 28, 28)
    x_train_e = np.array([np.fliplr(np.rot90(img, k=3)) for img in x_train_e])

    # Process Test EMNIST
    y_test_e = test_emnist.iloc[:, 0].values
    x_test_e = test_emnist.iloc[:, 1:].values.reshape(-1, 28, 28)
    x_test_e = np.array([np.fliplr(np.rot90(img, k=3)) for img in x_test_e])

    # --- 3. Merge ---
    x_train_final = np.concatenate((x_train_m, x_train_e), axis=0)
    y_train_final = np.concatenate((y_train_m, y_train_e), axis=0)
    
    x_test_final = np.concatenate((x_test_m, x_test_e), axis=0)
    y_test_final = np.concatenate((y_test_m, y_test_e), axis=0)

    # --- 4. Save to npy file for future use ---
    np.save('src/model/x_train_full.npy', x_train_final)
    np.save('src/model/y_train_full.npy', y_train_final)
    np.save('src/model/x_test_full.npy', x_test_final)
    np.save('src/model/y_test_full.npy', y_test_final)
    
    print("Finish saving 4 npy file. Total:", x_train_final.shape[0], "training data.")

save_merged_data('src/model/emnist-digits-train.csv', 'src/model/emnist-digits-test.csv')