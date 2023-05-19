import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")

rawData = pd.read_csv('dataset/crxdata.csv')
data = rawData[['A2', 'A3', 'A16']]
data_A, data_B = data.drop('A16', axis=1), data['A16']

fig, ax = plt.subplots()

ax.scatter(data[data['A16'] == 1]['A2'], data[data['A16'] == 1]['A3'], c='red', marker='+', label="+")
ax.scatter(data[data['A16'] == 0]['A2'], data[data['A16'] == 0]['A3'], c='blue', marker='o', label="-")
ax.set_xlabel('A2')
ax.set_ylabel('A3')

ax.set_title('Dataset Visualization')
ax.legend()

# mengambil data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(data_A, data_B, test_size=0.2, random_state=1)

def delete_multiple_lines(n=1):
    """Menghapus Baris Terakhir di STDOUT."""
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # kursor naik satu baris
        sys.stdout.write("\x1b[2K")  # hapus  baris terakhir

#perhitungan jarak euclidean
def naive_euclidian_distance(point1, point2):
    differences = [point1[x] - point2[x] for x in range(len(point1))]
    differences_squared = [difference ** 2 for difference in differences]
    sum_of_squares = sum(differences_squared)
    return sum_of_squares ** 0.5

# memulai waktu perhitungan
start = time.time()

#klasifikasi knn
def knn(x_train, y_train, x_test, actual=pd.DataFrame(), k=3, mode='test'):
    y_result = pd.DataFrame(columns=['y_pred'])
    score = 0

    if (x_train.shape == x_test.shape):
        raise Exception(f'The shape size is not same. x_train shape: {x_train.shape}, y_train: {y_train.shape}')
        return

    vertical_length_train, vertical_length_test = x_train.shape[0], x_test.shape[0]
    if (mode=='test'):
        for i in range(vertical_length_test):
            result_child = pd.DataFrame(columns=['distance', 'y'])
            for j in range(vertical_length_train):
                distance = naive_euclidian_distance(x_train.iloc[j], x_test.iloc[i])
                # menambah jarak dan y ke result_child, jangan gunakan append karena sudah usang
                result_child.loc[j] = [distance, y_train.iloc[j]]

                # print waktu
                print(f'{i+1}/{vertical_length_test}')
                # cetak print di terminal
                delete_multiple_lines(1)
            # dapatkan k jarak terdekat
            result_child = result_child.sort_values(by='distance')
            result_child = result_child.head(k)
            # dapatkan y_pred
            y_pred = result_child['y'].mode()[0]
            # tambahkan y_pred ke hasil
            y_result.loc[i] = [y_pred]
        print(y_result)

        # mendapatkan akurasi dari actual
        if (actual.shape[0] == y_result.shape[0]):
            for i in range(vertical_length_test):
                if (actual.iloc[i] == y_result.iloc[i][0]):
                    score += 1
            score = score / vertical_length_test

        return y_result, score
    else:
        raise Exception(f'Mode {mode} is not supported')
    
# mengerjakan perhitungan knn dengan k = 3, 5, 7
k_list = [3, 5, 7]
for k in k_list:
    print(f'K: {k}')
    y_result, score = knn(X_train, y_train, X_test, y_test, k=k, mode='test')
    print(f'Score: {score}')
    print(f'Time: {time.time() - start}')

    # menyimpan data ke dalam bentuk csv
    y_result.to_csv(f'result/knn_k_{k}.csv', index=False)

    # hasilkan plot dan simpan ke dalam png
    fig, ax = plt.subplots(nrows=2, sharey=True)
    ax[0].scatter(X_train[y_train == 1]['A2'], X_train[y_train == 1]['A3'], c='red', marker='+', label="+")
    ax[0].scatter(X_train[y_train == 0]['A2'], X_train[y_train == 0]['A3'], c='blue', marker='o', label="-")
    ax[0].set_xlabel('A2')
    ax[0].set_ylabel('A3')
    ax[0].set_title(f'Dataset Visualization with K = {k}')
    ax[0].legend()
    
    ax[1].scatter(X_test[y_test == 1]['A2'], X_test[y_test == 1]['A3'], c='red', marker='+', label="+")
    ax[1].scatter(X_test[y_test == 0]['A2'], X_test[y_test == 0]['A3'], c='blue', marker='o', label="-")
    ax[1].set_xlabel('A2')
    ax[1].set_ylabel('A3')
    ax[1].set_title(f'Dataset Visualization with K = {k}')
    ax[1].legend()
    plt.savefig(f'result/knn_k_{k}.png')

    print('\n')
    start = time.time()