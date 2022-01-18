# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#read data
dataset=pd.read_csv("Salary_Levels.csv")

#data frame
data=pd.DataFrame(dataset)

#independant and dependant
x=data[["level_id"]].astype(int)
y=data["Salary"].astype(int)

#ployfeatures and fitting model
poly=PolynomialFeatures(degree=3)
x_poly=poly.fit_transform(x)
polyreg=LinearRegression()
polyreg.fit(x_poly,y)
#prediction of L9 and L10
#1 is L3 so 8 is L10
L9=(polyreg.predict(poly.fit_transform([[7]])))
L10=(polyreg.predict(poly.fit_transform([[8]])))

#plot
#original points
plt.scatter(x,y,color='r',s=10)
#L9 prediction
plt.scatter(7,L9, color="green",s=10,marker="x")
#L10 predicition
plt.scatter(8,L10, color="green",s=10,marker="x")
#regression line
plt.plot(x,polyreg.predict(poly.fit_transform(x)),color='blue')
#axis
positions = (1, 2, 3,4,5,6,7,8)
labels = ("L3","L4","L5","L6","L7","L8","L9","L10")
plt.xticks(positions, labels)
#labels
plt.title("Google SWE Salary Regression")
plt.ticklabel_format(style='plain', axis='y')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
