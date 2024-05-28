import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cars_processed.csv')
#print(df.info())
plt.figure(1)

# razliciti prikazi
sns.pairplot(df, hue='fuel')

sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
df = df.drop(['name','mileage'], axis=1)

obj_cols = df.select_dtypes(object).columns.values.tolist()
num_cols = df.select_dtypes(np.number).columns.values.tolist()

fig = plt.figure(figsize=[15,8])
for col in range(len(obj_cols)):
    plt.subplot(2,2,col+1)
    sns.countplot(x=obj_cols[col], data=df)

df.boxplot(column =['selling_price'], grid = False)
df.hist(['selling_price'], grid = False)
tabcorr = df.drop(['fuel', 'seller_type', 'transmission', 'owner'], axis=1).corr()
sns.heatmap(df.drop(['fuel', 'seller_type', 'transmission', 'owner'], axis=1).corr(), annot=True, linewidths=2, cmap= 'coolwarm')

plt.show()


#1.Koliko mjerenja (automobila) je dostupno u datasetu? 6699
#2.Kakav je tip pojedinog stupca u dataframeu?
        # 0   name               object
        # 1   year              int64
        # 2   selling_price     float64
        # 3   km_driven         int64
        # 4   fuel              object
        # 5   seller_type       object
        # 6   transmission      object
        # 7   owner             object
        # 8   mileage           float64
        # 9   engine            int64
        # 10  max_power         float64
        # 11  seats             int64
#3.Koji automobil ima najveću cijenu, a koji najmanju? najmaji: 10.30
#4.
#5.
#6.
#7.
