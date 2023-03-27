import copy
import numpy as np
import pandas as pd


def gaussian_noise(x, mu, std):
    noise = np.random.normal(mu, std, size=x.shape)
    x_noisy = x + noise
    return x_noisy


def temp_detector_data2(dataset_path):
    df = pd.read_csv(dataset_path, index_col='_time', parse_dates=True)
    data = df[["J1", "J2", "J3", "J4", "J5", "J6"]]
    window_size = 288
    num_samples = len(data)
    num_windows = num_samples // window_size
    data = data.iloc[:num_windows * window_size]  # Drop any extra samples at the end

    x_train = data.values.reshape(num_windows, window_size, -1)

    x_test = copy.deepcopy(x_train)
    y_test = []     # 0/1
    y_class_test = []       # 0/1/2/3/4

    for i in range(len(x_test)):
        window = x_test[i]
        anomaly_type = np.random.randint(0, 5)

        # No anomaly keep unchanged
        if anomaly_type == 0:
            y_test.append(0)
            y_class_test.append(0)

        # Gaussian noise anomaly
        elif anomaly_type == 1:
            y_test.append(1)
            y_class_test.append(1)
            feature_idx = np.random.randint(6)
            noise = np.random.normal(loc=0, scale=0.2, size=window_size)
            window[:, feature_idx] += noise

        # Invert values of features
        elif anomaly_type == 2:
            y_test.append(1)
            y_class_test.append(2)
            feature_idx = np.random.randint(6)
            window[:, feature_idx] *= -1

        # Add spike
        elif anomaly_type == 3:
            y_test.append(1)
            y_class_test.append(3)
            feature_idx = np.random.randint(6)
            spike_pos = np.random.randint(window_size)
            window[spike_pos, feature_idx] += np.random.normal(loc=2, scale=0.1)

        # Uniform noise
        else:
            y_test.append(1)
            y_class_test.append(4)
            feature_idx = np.random.randint(6)
            noise = np.random.uniform(low=-1, high=1, size=window_size)
            window[:, feature_idx] += noise

        x_test[i] = window

    counts = np.bincount(y_class_test)

    print('Length x_train : ' ,len(x_train))
    print('Length x_test  :' ,len(x_test))
    # Print the counts for each value
    for i, count in enumerate(counts):
        print(f"Class {i} : {count} ")

    # x_train, y_train, x_test, y_test, y_test_classes
    return x_train, x_train, x_test, np.array(y_test), np.array(y_class_test)


def temp_detector_data(dataset_path):
    df = pd.read_csv(dataset_path, index_col='_time', parse_dates=True)
    df_len = round(len(df) - 1)
    data = df[["J1", "J2", "J3", "J4", "J5", "J6"]].head(df_len)
    data = data.values.tolist()
    size = 288
    step = 96
    x_train = np.array([data[i:(i + size)] for i in range(0, len(data) - size + 1, step)])
    x_test = copy.deepcopy(x_train)

    for i in range(round(len(x_test)/2)):
        mu = 0.0
        std = 0.3 * np.std(x_test[i])
        x_test[i] = gaussian_noise(x_test[i], mu, std)

    y_test = np.concatenate([np.ones(round(len(x_test)/2)), np.zeros(len(x_test) - round(len(x_test)/2))])

    return x_train, x_train, x_test, y_test


def temp_classifier_data():
    # Define the number of classes
    num_classes = 5

    # Define the shape of the windows
    window_shape = (288, 6)

    # Create an empty numpy array to store the windows
    x_train = np.empty((num_classes * 20, *window_shape))

    # Create an empty numpy array to store the labels
    y_train = np.empty(num_classes * 20, dtype=int)

    # Generate 20 unique windows for each class
    for class_label in range(num_classes):
        for i in range(20):
            # Generate a random window with random values between 0 and 1
            window = np.random.rand(*window_shape) * 5

            # Add a different pattern to windows labeled with 1
            if class_label == 0:
                window += np.array([1, 0, 0, 0, 0, 0])
            elif class_label == 1:
                window += np.array([0, 1, 0, 0, 0, 0])
            elif class_label == 2:
                window += np.array([0, 0, 1, 0, 0, 0])
            elif class_label == 3:
                window += np.array([0, 0, 0, 1, 0, 0])
            elif class_label == 4:
                window += np.array([0, 0, 0, 0, 1, 0])

            # Add the window to the x_train array
            x_train[class_label * 20 + i] = window

            # Add the label to the y_train array
            y_train[class_label * 20 + i] = class_label

    return x_train, y_train, x_train, y_train