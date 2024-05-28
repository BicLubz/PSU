import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")
print(img.shape)
print(img.dtype)

img_copy = img[:,:,0].copy()
broj = 150
brightnes = 255 - broj #inkrementiran broj se dodaje svim pikselima slike
#brightnes = np.clip(brightnes, 0, 255) #vrijednost brightnes mora biti unutar granica rgb 0-255

for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img_copy[x][y] > brightnes:
                img_copy[x][y] = 255
            else:
                img_copy[x][y] += brightnes

mirror = img[:, :: -1] #zrcaljena slika, rezanje polja slike, prvi :, oznacava sve sve retke slike a :: -1 oznacava sve stupce u obrnutom redosljedu (invertiramo redosljed piksela u svakom retku

rotacija = np.rot90(img, k = 1, axes=(1,0)) #jedna rotacija za 90 stupnjeva suprotno od smjera kazaljke na satu

length = img.shape[0]
height = img.shape[1]
reshaped = img.reshape(img.shape[0], img.shape[1]).mean(axis = (1, 3))



plt.figure()
plt.imshow(img, cmap="gray")
plt.title("original")

plt.figure() #prikaz slike
plt.imshow(img_copy, cmap="gray") #prikazuje sliku koristeci mapu boja u sivoj razini
plt.title("brightnes") #naslov slike

plt.figure()
plt.imshow(rotacija, cmap="gray")
plt.title("rotirana")

plt.figure()
plt.imshow(mirror, cmap="gray")
plt.title("zrcaljena slika")

plt.figure()
plt.imshow(reshaped, cmap="gray")
plt.title("Smanjena rezolucija")


druga_cetvrtina = img[0 : img.shape[0], img.shape[1] // 4 : img.shape[1] // 2].copy()
