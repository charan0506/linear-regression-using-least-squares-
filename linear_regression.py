# linear-regression-using-least-squares-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("/home/charan/Downloads/Deep-Learning-Linear-Regression-master/data.csv")
x,y=df['x'],df['y']
x_mean=x.mean()

y_mean=y.mean()

b_1=sum((x-x_mean)*(y-y_mean))/sum((x-x_mean)**2)

b_0=y_mean-b_1*x_mean

plt.scatter(x,y)
y_predit=b_1*x+b_0
plt.plot(x,y_predit)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
