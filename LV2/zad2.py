import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1, 2, 3, 4, 5, 6), delimiter=",", skiprows=1)

mpg = data[:, 0]  # 0 indeks jer je u usecols ignoriran prvi stupac u tablici
hp = data[:, 3]
tezina = data[:, 5]
cyl = data[:, 1]

mpg_min = min(mpg)
mpg_max = max(mpg)
mpg_avg = np.mean(mpg)

plt.xlabel('mpg')
plt.ylabel('hp')

mpg6 = mpg[np.where(cyl == 6)[0]]  #potrosnja auta koji imaju samo 6 cilindara
mpg6_min = min(mpg6)
mpg6_max = max(mpg6)
mpg6_avg = np.mean(mpg6)

plt.scatter(mpg, hp, linewidth=tezina)#crtanje grafa sa mpg na x i hp na y osi sa točkama čija je debljina određena težinom auta
plt.show()

print(mpg_min)
print(mpg_max)
print(mpg_avg)

print(mpg6_min)
print(mpg6_max)
print(mpg6_avg)
