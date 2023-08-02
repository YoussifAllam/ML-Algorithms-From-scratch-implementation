import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path=r'F:1.TXT'
data = pd.read_csv( path , header = None , names = ['Population', 'Profit'] )



print('data = \n' ,data.head(10) )
print('**************************************')
print(data.describe())

data.plot( kind="scatter" , x='Population' , y='Profit' ,figsize=(5,5) )

data.insert(0,'Ones',1)
print('**************************************')
print(data)


cols=data.shape[1] 
print(cols)

#! separate X (training data) from y (target variable)
X = data.iloc[ : , 0 : cols - 1 ]  # select all rows and  from col 0 and  1 
y = data.iloc[ : , cols - 1 : cols]


#! convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values) 
theta = np.matrix(np.array([0,0])) # theta 0 = 0 ,thete1 = 0

print('**************************************')
print(X,"\n",X.shape) # ----> 97, 2
print('**************************************')
print(y,"\n",y.shape)


#! cost function  
#عشان احسب قيمها الخطأ  
def cost_function(X,y,theta):
    z = np.power( ( ( X*theta.T ) - y ) , 2 )  # H_theta(X) = (X*theta.T ) (97,2)*(2*1)
    return sum(z) / ( 2 * len(X) )

print(cost_function(X,y,theta))







#! gradientDescent 
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1]) 


    cost = np.zeros(iters)

    for i in range(iters): 
        error = (X * theta.T) - y 
        
        for j in range(parameters): 
            term = np.multiply(error, X[:,j]) 

            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp # update theta in each iteration 

        cost[i] = cost_function(X, y, theta) 
        
    return theta, cost







#! initialize variables for learning rate and iterations
alpha = 0.01
iters = 1000

#! perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)

print('g = ' , g)
print('cost  = ' , cost[0:50] )
print('computeCost = ' , cost_function(X, y, g))
print('**************************************')



x = np.linspace(data.Population.min(), data.Population.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x) 



#! draw the line

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


#! draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()



