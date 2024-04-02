import pandas as pd
import numpy as np
mtcars = pd.read_csv('mtcars.csv')
new_mtcars = mtcars.sort_values(by='mpg',ascending = False).head(5)
print("5 auta koji imaju najvecu potrosnju su:")
print(new_mtcars[['car','mpg']])

print("##########################################################################")

new_mtcars = mtcars.sort_values(by='mpg',ascending=True).head(3)
print("3 auta sa 8 cilindara koji trose najmanje:")
print(new_mtcars[new_mtcars.cyl == 8])
