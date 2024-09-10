import csv
import numpy as np

data = np.load('../Features/Pre-trained_features/positive_tape_bert.npz', allow_pickle=True)
tmp = data.files

for i in range(len(tmp)):
    train_P = data[tmp[i]]
    dictp = train_P.tolist()
    dict1 = dict(dictp)
    des = dict1['pooled']
    pooled_list = des.tolist()
    f = open("../Features/Feature_csv/positive_tape_bert.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(pooled_list)
    f.close()

import pandas as pd
file_path = '../Features/Feature_csv/positive_tape_bert.csv'
df = pd.read_csv(file_path, header=None)

headers = list(range(768))
df.columns = headers
output_path = '../Features/Feature_csv/positive_tape_bert.csv'
df.to_csv(output_path, index=False)