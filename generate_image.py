import numpy as np
import utility as ut
import pandas as pd
from sklearn import preprocessing
import argparse

def dec_to_bin(x):
    return format(int(x), "b")

def rgb_image_generation(df):
    list_image_flat = []
    size, padding = get_image_size(num_col)
    vec = [[0, 0, 0]] * padding
    for (index_label, row_series) in df.iterrows():
        list_image = []
        j = 0
        while j < len(row_series):
            if row_series[j] > 1:
                c = 1.0
            elif row_series[j] < 0:
                c = 0.0
            else:
                c = row_series[j]
            v = c * (2 ** 24 - 1)
            bin_num = dec_to_bin(int(v))
            if len(bin_num) < 24:
                pad = 24 - len(bin_num)
                zero_pad = "0" * pad
                line = zero_pad + str(bin_num)
                n = 8
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
                int_num = [int(element, 2) for element in rgb]
            else:
                n = 8
                line = str(bin_num)
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
                int_num = [int(element, 2) for element in rgb]
            list_image.append(int_num)
            j = j + 1
        list_image = list_image + vec
        new_img = np.asarray(list_image)
        new_img = new_img.reshape(size, size, 3)
        list_image_flat.append(new_img)
    return list_image_flat

def get_image_size(num_col):
    import math
    matx = round(math.sqrt(num_col))
    if num_col>(matx*matx):
        matx = matx + 1
        padding = (matx*matx) - num_col
    else:
        padding = (matx*matx) - num_col
    return matx, padding

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Inception for next activity prediction.')
        parser.add_argument('-event_log', type=str, help="Event log name")
        args = parser.parse_args()
        namedataset = args.event_log
        namedataset = namedataset
        df = pd.read_csv('kometa_fold/'+namedataset+'feature.csv', header=None)
        fold1, fold2, fold3 = ut.get_size_fold(namedataset)
        df = df.iloc[:, :-1]
        num_col = len(df. columns)
        X_1 = df[:fold1]
        X_2 = df[fold1:(fold1+fold2)]
        X_3 = df[(fold1+fold2):]

        f = 0

        for f in range(3):
            print("Fold n.", f)
            if f == 0:
                X_train = X_1.append(X_2)
                X_test = X_3
            elif f == 1:
                X_train = X_2.append(X_3)
                X_test = X_1
            elif f == 2:
                X_train = X_1.append(X_3)
                X_test = X_2

            dataframe_train = X_train
            dataframe_test = X_test

            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            scaler.fit(dataframe_train.values.astype(float))

            train_norm = scaler.transform(dataframe_train.values.astype(float))
            test_norm = scaler.transform(dataframe_test.values.astype(float))

            train_norm = pd.DataFrame(train_norm)
            test_norm = pd.DataFrame(test_norm)

            print("prepare fold n.", f)
            X_train = rgb_image_generation(train_norm)
            X_test = rgb_image_generation(test_norm)

            np.save("image/"+namedataset+"/"+namedataset+"_train_fold_"+str(f) + ".npy", X_train)
            np.save("image/"+namedataset+"/"+namedataset+"_test_fold_" + str(f) + ".npy", X_test)

