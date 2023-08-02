import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r'F:\1.TXT'
data = pd.read_csv( path, header=0 , names=['Exam 1', 'Exam 2', 'Admitted'])
#?print(data.head(10))


positive = data[ data['Admitted'].isin([1]) ]
negative  = data[ data['Admitted'].isin([0]) ]
#? print(negative)



#! draw


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

#! sigmoid

def sigmoid(z):
    return 1 / (1 + np.exp(-z))



#! cost fun
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first_term = np.multiply(-y, np.log(sigmoid(X * theta.T))) # at y = 1
    second_term = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T))) # at y =0
    return np.sum(first_term - second_term ) / (len(X))



#! add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)


#! set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


#! convert to numpy matrix and initalize the parameter array theta    
X = np.array(X.values)
y = np.array(y.values)
#theta = np.zeros(3)
theta = np.matrix(np.array([0,0,0]))

#? print("theta shape ",theta.shape) # shape = 1 * 3

thiscost = cost(theta, X, y)
print('cost = ' , thiscost)


#! grad fun
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters) 
    
    error = sigmoid(X * theta.T) - y 
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad


import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y)) # to get best value to thetas
#? print()
print("res = ",result)



costafteroptimize = cost(result[0], X, y)
print()
print('cost after optimize = ' , costafteroptimize)
print()



def predict(theta, X): 
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X) 
#? print(predictions) # el qyam el motoq3a 

correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]



accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))




















