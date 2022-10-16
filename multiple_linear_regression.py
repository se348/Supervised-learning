import math, copy
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])


def predict(x, w, b): 
    p = np.dot(x, w) + b    
    return p   


def compute_cost(X,y,w,b):
    no_of_train= np.shape(X)[0]
    no_of_features= np.shape(X[0])[0]
    total_cost=0
    for i in range(no_of_train):
        total_cost += (predict(X[i],w,b) - y[i])**2
    return total_cost/ (2 * no_of_train)

def compute_gradient(X,y,w,b):
    m,n =np.shape(X)
    grad_n= np.zeros(n)
    grad_b=0
    for i in range(m):
        diff =predict(X[i],w,b) -y[i]
        for j in range(n):
            grad_n[j] += diff * X[i][j]
        grad_b += diff
    return grad_n/m, grad_b/m 

       
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in) 
    b = b_in
    
    for i in range(num_iters):

        dj_db,dj_dw = gradient_function(X, y, w, b)

        b = b - alpha * dj_db           
        w = w - alpha * dj_dw           
      
        if i<100000:   
            J_history.append( cost_function(X, y, w, b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history 