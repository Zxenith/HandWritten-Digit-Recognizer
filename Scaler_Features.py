import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Datasets/emnist-reshaped-bymerge.csv')
df = df.iloc[:, 1:]

scaler = StandardScaler()
scaler.fit(df)

meany = scaler.mean_
standy = scaler.scale_

with open("mean.txt", "w") as f1:
    f1.write(str(meany))
    f1.close()

with open("standy.txt", "w") as f2:
    f2.write(str(standy))
    f2.close()