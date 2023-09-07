import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path= r"F:\2.TXT"
data = pd.read_csv(path , header=None ,names=['Size', 'Bedrooms', 'Price'])


def computeCost(X,y,Theta):
    z = np.power((X*Theta.T)-y,2)
    return np.sum(z/(2*len(X))) 

#! data rescaling

data = (data- data.mean())/data.std()

 

data.insert(0, 'Ones', 1) # Xo




cols = data.shape[1] # = 3
X2 = data.iloc[:,0:cols-1] 
y2 = data.iloc[:,cols-1:cols] 




#! convert to matrices and initialize theta
X2 = np.matrix(X2.values)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",X2.shape)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))




def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost



# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

# perform linear regression on the data set
new_theta, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
thiscost = computeCost(X2, y2, new_theta)


print('g2 = ' , new_theta)
print('**************************************')
print('cost2  = ' , cost2[0:1000] )
print('**************************************')
print('computeCost = ' , thiscost)


#! get best fit line for Size vs. Price
x = np.linspace(data.Size.min(), data.Size.max(), 100) # data.Size.min() to get the smallest value in size col
print('x \n',x)
print('g \n',new_theta)

f = new_theta[0, 0] + (new_theta[0, 1] * x)
print('f \n',f)


#! draw the line for Size vs. Price
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Size, data.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')



#! get best fit line for Bedrooms vs. Price
x = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)  # get the smallest value in bedrooms col
print('x \n',x)
print('g \n',new_theta)

f = new_theta[0, 0] + (new_theta[0, 1] * x)
print('f \n',f)



#! draw the line  for Bedrooms vs. Price
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Bedrooms, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')




#! draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')




#! 3D

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(data.Size.min(), data.Size.max(), 100)
ax.set_xlabel('Time')

y = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100) 
ax.set_ylabel('Bedrooms')

z = np.linspace(data.Price.min(), data.Price.max(), 100)
ax.set_zlabel('Price')


ax.plot3D(x, y, z, 'b') 



plt.show()

