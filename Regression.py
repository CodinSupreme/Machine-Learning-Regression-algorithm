import matplotlib.pyplot as plt
import numpy as np
import math
import random


class Simple_Linear_Regression:
    def __init__(self) -> None:
        self.prediction = None

    def Cardacian_coefficient(self, x, y):
        
        xmean = np.mean(x)
        ymean = np.mean(y)

        x= np.array(x) 
        y= np.array(y)

        r = np.dot((x - xmean), (y - ymean))/ math.sqrt(np.dot((x-xmean), (x - xmean)) * np.dot((y-xmean), (y - ymean)))
        return r
    
    def stdev(self, m):
        sum = 0
        mean = np.mean(m)
        n = len(m)
        for i in m:
            sum += math.pow((i - mean), 2)

        return math.sqrt(sum/(n - 1))

    def Slope(self, x, y, r):
        return r * self.stdev(y)/self.stdev(x)

    def Intercept(self, X, Y, m):
        sumx = 0
        sumy = 0
        for x,y in zip(X, Y):
            sumx += x 
            sumy += y 
        
        return (sumy - m * sumx)/len(Y)

    def Fit(self, x, y):
        self.r = self.Cardacian_coefficient(x, y)
        self.slope = self.Slope(x, y, self.r) 

        self.intercept = self.Intercept(x, y, self.slope)

    def Predict(self, x):
        self.prediction = self.slope*x + self.intercept
        return self.prediction

class Logical_Regression:
    def __init__(self) -> None:
        self.prediction = None
        self.intercept = None

    def Fit(self, x, y, n = 1):
        if n == None:
            self.n = 3
        else:
            self.n = n

        temp = [] 
        for i in x:
            lst = []
            for j in range(self.n+1):
                lst.append(i ** j)
            
            temp.append(lst)

        x = temp.copy()


        
        x_T = np.mat(np.transpose(x))
        x = np.mat(x)
        y = np.mat(np.transpose([y]))


        result = np.mat(np.linalg.inv(x_T * x)) * x_T * y

        self.b = list(np.array(np.transpose(result))[0])
        
        self.intercept = float(self.b[0])
        self.b.pop(0)

    def Predict(self, x):
        temp = [] 
        for j in range(1, self.n+1):
            temp.append(x ** j)

        x = temp.copy()

        result = round(np.dot(self.b, x) + self.intercept, 3)

        self.prediction = round(1/(1 + math.exp(- result)) , 0)

        return self.prediction

class Polynomial_Regression:
    def __init__(self) -> None:
        self.prediction = None
        self.intercept = None

    def Fit(self, x, y, n = None):
        if n == None:
            self.n = 3
        else:
            self.n = n

        temp = [] 
        for i in x:
            lst = []
            for j in range(self.n+1):
                lst.append(i ** j)
            
            temp.append(lst)

        x = temp.copy()

        x_T = np.mat(np.transpose(x))
        x = np.mat(x)
        y = np.mat(np.transpose([y]))

        result = np.mat(np.linalg.inv(x_T * x)) * x_T * y

        self.b = list(np.array(np.transpose(result))[0])
        
        self.intercept = float(self.b[0])
        self.b.pop(0)

    def Predict(self, x):
        temp = [] 
        for j in range(1, self.n+1):
            temp.append(x ** j)

        x = temp.copy()

        self.prediction = round(np.dot(self.b, x) + self.intercept, 3)

        return self.prediction
            
class Multilinear_regression:
    def __init__(self) -> None:
        self.prediction = None
        self.intercept = None

    def Fit(self, x, y):
        for i in x:
            print(i)
            i.insert(0, 1)

        
        x_T = np.mat(np.transpose(x))
        x = np.mat(x)
        y = np.mat(np.transpose([y]))

        print(np.shape(x), np.shape(y))
        print(x)
        print(y)

        result = np.mat(np.linalg.inv(x_T * x)) * x_T * y

        self.b = list(np.array(np.transpose(result))[0])
        
        self.intercept = float(self.b[0])
        self.b.pop(0)

    def Predict(self, x):
        self.prediction = round(np.dot(self.b, x) + self.intercept, 3)
        return self.prediction

def func(n):
    x = []
    y = []

    for i in range(n):
        y.append(i)
        temp = []
        for j in range(1, n):
            temp.append(i ** j)

        x.append(temp)

    return x, y

def func2(n, r):
    x= []
    y= []

    for i in range(n):
        j = random.randrange(0, r)
        while j in x:
            j = random.randrange(0, r)
        
        x.append(j)
        y.append(random.randrange(0, 2))

    return x, y

def main():
    x, y = func(7)
    x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
    y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

    x, y = func2(20, 100)
    x = [1,2, 3, 4, 5, 10, 11, 12, 13, 14]
    y=[0, 0, 0, 0 ,0, 1, 1, 1, 1, 1]
    x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
    y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

    reg =  Polynomial_Regression()
    reg.Fit(x, y)

    #x, y = func(7)

    n =3

    reg.Predict(x[n])
    print("#_________Results___________#")
    print(f"Prediction: {reg.prediction}")
    print(f"Actual: {y[n]}")

    result = []
    for i in range(30):
        result.append(reg.Predict(i))
    
    plt.scatter(x, y)
    plt.plot(result)
    plt.show()

main()

