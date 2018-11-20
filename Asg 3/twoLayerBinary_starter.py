'''
This file implements a two layer neural network for a binary classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb

def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    '''
    computes derivative of tanh activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    return dZ

def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE - DONE
    # A = sigma(Z)
    # dL/dz = dA/dZ = dL/dA2 * A2* (1-A2)
    
    A = cache["ACur"]
    # print A.shape
    dZ = dA*A*(1-A)
    # print dZ.shape
    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):
    '''
    Initializes the weights of the 2 layer network

    Inputs: 
        n_in input dimensions (first layer)
        n_h hidden layer dimensions
        n_fin final layer dimensions

    Returns:
        dictionary of parameters
    '''
    # initialize network parameters
    ### CODE HERE -- DONE
    # since variance is 1 which gives larger weights for random.rand  we multiply with 0.01
    # W nout*nin
    W1 = np.random.rand(n_h,784)*0.01
    b1 = np.random.rand(n_h,1)*0.01

    W2 = np.random.rand(1,n_h)*0.01
    b2 = np.random.rand(1,1)*0.01

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A, W and b
        to be used for derivative
    '''

    cache = {}
    cache["APrev"] = A

    ### CODE HERE -- DONE

    # print W.shape, A.shape
    Z = np.dot(W,A) + b
    cache["W"]=W
    cache["b"]=b    
    # print Z.shape
    # dimensions
    # Z1 - (500*784)*(784*2000) = 500*2000
    # A1=Z1
    # Z2 - (1*500)*(500*2000) = 1*2000
    # A2=Z2
    
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    #lin_cache will have Aprev and Acure both since these values are used in derivative

    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)

    # Current A
    lin_cache["ACur"] = A

    # activation cache is only [{"Z" :Z}]   
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache

def cost_estimate(A2, Y):
    '''
    Estimates the cost with prediction A2

    Inputs:
        A2 - numpy.ndarray (1,m) of activations from the last layer
        Y - numpy.ndarray (1,m) of labels
    
    Returns:
        cost of the objective function
    '''
    ### CODE HERE - DONE
    # cost = -(yloga  + (1-y)log(1-a))
   
    m = Y.shape[1]
 
    cost = np.sum(Y*np.log(A2)+ (1-Y)*np.log(1-A2))
    cost= -1*cost/m;    
    return cost

def linear_backward(dZ, cache):
    '''
    Backward propagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs Aprev,W,b,Acur
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    # CODE HERE - DONE
    W = cache["W"]
    b = cache["b"]

    A_prev = cache["APrev"]
    # print dZ.shape,W.shape,A_prev.shape

    dA_prev = np.dot(W.T,dZ)
    dW = np.dot(dZ,A_prev.T)
    db = np.sum(dZ,axis=1,keepdims=1)

    return dA_prev, dW, db

def layer_backward(dA, cache, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''

    # dA2 = 1*2000
    # dA1 = 500*2000
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]
    
    # get dz from here
    # dz is da/dz which is dl/dz

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, lin_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, lin_cache)
    
    # then we compute dz/dw and dz/db which also gives dAprev since
    #  Z = WAprev + b
    # dZ2 -> same as dA2 -> 1*2000
    # dZ1 -> same as dA1 -> 500*2000
     
    # print dZ.shape

    dA_prev, dW, db = linear_backward(dZ, lin_cache)

    return dA_prev, dW, db

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    A0 = X
    A1,cache1 = layer_forward(A0, W1, b1, "sigmoid")   
    A2,cache2 = layer_forward(A1, W2, b2, "sigmoid")
    YPred =[]
    a2=A2.tolist()
  
    for i in a2[0]:
       
        if(i>=0.5):
            YPred.append(1)
        else:
            YPred.append(0)
  
    return YPred

def two_layer_network(X, Y, net_dims,VData,VLabel,num_iterations=1000, learning_rate=0.1):
    '''
    Creates the 2 layer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)

    # first Activation is X
    A0 = X
    A3 = VData

    costs = []
    vcosts=[]
    mainCache=[]

    # list of dictionaries of A and Z
    # [{"A":A0,"Z":Z1},{"A":A1,"Z":Z2}]
    
      
    
    for ii in range(num_iterations+1):
        # Forward propagation -  2 layers
        ### CODE HERE - DONE
        W1=parameters["W1"]
        b1=parameters["b1"]
        W2=parameters["W2"]
        b2=parameters["b2"]
        A1,cache1 = layer_forward(A0, W1, b1, "sigmoid")
        
        # cache1 - 1) lin_cache  - {"A":A0,"W":W1,"b":b1}
        #          2) act_cache  - {"Z":Z1}
        
        A2,cache2 = layer_forward(A1, W2, b2, "sigmoid")
        # cache2 - 1) lin_cache  - {"A":A1,"W":W2,"b":b2}
        #          2) act_cache  - {"Z":Z2}


        # validation data
        A4,cache3 = layer_forward(A3,W1,b1,"sigmoid")
        A5,cache4 = layer_forward(A4,W2,b2,"sigmoid")

        # cost estimation
        ### CODE HERE - DONE
        # training cost
        cost = cost_estimate(A2,Y)

        # validation cost
        vcost = cost_estimate(A5,VLabel)
     
        
        # Backward Propagation
        ### CODE HERE  - DONE
        # calculate dA2
        # dA2 = (dL/dA) = derivate if sum(Y*log(A)+ (1-Y)*log(1-A))
        # dA, cache, W, b, activation

        m = Y.shape[1]
        dA2 = (Y/A2 + (1-Y)/(A2-1))
        dA2 = -1*(dA2/m)
        # dA2 -> 1*2000          
        
        # print "in 2 layer back propagation"
        dA1, dW2, db2=layer_backward(dA2,cache2,"sigmoid")
       
       
        # print "in 1 layer back propagation"
        dA0, dW1, db1=layer_backward(dA1,cache1,"sigmoid")
        
        #update parameters
        ### CODE HERE - DONE
      
        parameters["W1"]=W1-learning_rate*dW1
        parameters["b1"]=b1-learning_rate*db1
        parameters["W2"]=W2-learning_rate*dW2
        parameters["b2"]=b2-learning_rate*db2

        if ii % 10 == 0:
            costs.append(cost)
            vcosts.append(vcost)
    
        if ii % 1000 == 0:
            print "Training Cost at iteration %i is: %f" %(ii, cost) 
            print "Validation Cost at iteration %i is: %f\n" %(ii, vcost)
                    
    
    return costs,vcosts, parameters,A2

def calculateAccuracy(Y,YPred):
    y=Y.tolist()[0]
    # YPred is a list already
    n = len(YPred)
    count=0
    for i in range(0,n):
        if(YPred[i]==y[i]):
            count+=1
    accuracy = (float(count)/float(n))*100
   
    return accuracy

def main():
    # getting the subset dataset from MNIST
    # binary classification for digits 1 and 7
    hiddenNodes = [100,200,500]
    digit_range = [1,7]
    data, label, test_data, test_label = \
            mnist(noTrSamples=2400,noTsSamples=1000,\
            digit_range=digit_range,\
            noTrPerClass=1200, noTsPerClass=500)
   
    train_data = data[:,0:1000]
    train_data2 = data[:,1400:2400]    
    train_data = np.column_stack((train_data,train_data2)) 
  
    
    train_label = label[:,0:1000]
    train_label2 = label[:,1400:2400]
    train_label = np.column_stack((train_label,train_label2)) 


    validation_data = data[:,1000:1400]
    validation_label = label[:,1000:1400]

    # intialially its 1 and 7
    #convert to binary labels
    #  0 -> 1
    #  1 -> 7
    train_label[train_label==digit_range[0]] = 0
    train_label[train_label==digit_range[1]] = 1
    
    test_label[test_label==digit_range[0]] = 0
    test_label[test_label==digit_range[1]] = 1
    
    validation_label[validation_label==digit_range[0]] = 0
    validation_label[validation_label==digit_range[1]] = 1
	
    mxValidationAccuracy=0
    testError=0
    nodes=200
    mxvcost=99

    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle('\nTraining cost v iterations\n')
    i=0
    for n_h in hiddenNodes:
        if(n_h==400):
            print "\nVarying hidden nodes to check accuracy"
        print "Number of hidden nodes :",n_h
       
        n_in, m = train_data.shape
        # n_in == 784 , m==2000
        n_fin = 1

        # for 500 nodes
        # NN ->  784->500_->1
        net_dims = [n_in, n_h, n_fin]
        # initialize learning rate and num_iterations
        learning_rate = 0.1
        num_iterations = 1000

        # print train_label
        costs,vcosts, parameters,A2 = two_layer_network(train_data, train_label, net_dims, \
                validation_data,validation_label,num_iterations=num_iterations, learning_rate=learning_rate)

        
        # compute the accuracy for training set and testing set
        train_Pred = classify(train_data, parameters)
        train_Accuracy = calculateAccuracy(train_label,train_Pred)
        
        validation_Pred =classify(validation_data,parameters)
        validation_Accuracy = calculateAccuracy(validation_label,validation_Pred)

        test_Pred = classify(test_data, parameters)
        test_Accuracy = calculateAccuracy(test_label,test_Pred)
        
        print "Accuracy for training set is ",train_Accuracy
        print "Accuracy for validation set is ",validation_Accuracy
        print "Accuracy for testing set is ",test_Accuracy
        
        print "Test Error is ",100 - test_Accuracy,"%","\n" 

        if(float(vcosts[100])<float(mxvcost)):
            mxvcost=vcosts[100]
            testError = test_Accuracy
            nodes=n_h
            mxValidationAccuracy=validation_Accuracy

        # CODE HERE TO PLOT costs vs iterations
        iterations=[k for k in range(0,1001,10)]
        axs[i].plot(iterations,costs,'r',label="Training Cost")
        axs[i].plot(iterations,vcosts,'b',label="Validation Cost")
        axs[i].set_title("Hidden nodes :"+ str(n_h))
        axs[i].set_xlabel('Iterations')
        axs[i].set_ylabel('Costs')
        axs[i].legend()
        
        i+=1
        # axs[1].plot(iterations,vcosts)
        # axs[1].set_xlabel('Iterations')
        # axs[1].set_title('\nValidation cost vs iterations\n"')
        # axs[1].set_ylabel('Costs')
        
    plt.show()
    print "Best Validation Cost is: ", mxvcost," where number of hidden nodes :",nodes
    print "Validation accuracy for this architecture :", mxValidationAccuracy
    print "Test Accuracy for this architecture  :", testError
    print "\n"

if __name__ == "__main__":
    main()




