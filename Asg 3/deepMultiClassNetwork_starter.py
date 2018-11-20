'''
This file implements a multi layer neural network for a multiclass classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast
np.set_printoptions(threshold=np.nan)

def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    cache["Z"]=Z
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''

    ### CODE HERE  - DONE
    # for the last layer we compute A again using softmax activation such that sum of all As is 1/
    # each A[i] gives probability of digit to be equal to i. 

    # np.max(Z) is subtracted to avoid nan value ( z large can cause overflow)
    
    # Softmax activation
    A=[]
    n = Z.shape[1]
   
    for i in range(0,n):
        exps = np.exp(Z[:,i] - np.max(Z[:,i]))
        A.append(exps / np.sum(exps))
        
    A = np.array(A,dtype=np.float)
    A = A.T

    cache={}
    cache["Alast"]=A

    # LOSS - cross entropy sum of -logak for all samples where yk=1
    loss = 0
    for j in range(0,n):
        for i in range(0,10):
            if Y[i][j]==1: 
                if(A[i][j]==0):
                    print i,j,A[:,j]
                loss+=(-1*np.log(A[i][j]))
                break
  
    loss = loss
    loss = loss/n
    
    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE 
    dZ= []
    A = cache["Alast"]
    n = Y.shape[1]
    
    for i in range(0,n):
        dZ.append((A[:,i]-Y[:,i])/n)
        # print dZ
    dZ = np.array(dZ,dtype=float)
    dZ = dZ.T

    return dZ

def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}

    for l in range(numLayers-1):
        n_out= net_dims[l+1]
        n_in = net_dims[l]
        parameters["W"+str(l+1)] = np.random.rand(n_out,n_in)*0.01
        parameters["b"+str(l+1)] = np.random.rand(n_out,1)*0.01

    # print parameters["W1"].shape,parameters["W2"].shape,parameters["b1"].shape,parameters["b2"].shape,parameters["W3"].shape,parameters["b3"].shape,
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
        cache - a dictionary containing the inputs A,W,b
    '''
    ### CODE HERE - DONE
    cache = {}
    cache["APrev"] = A

    Z = np.dot(W,A) + b
    cache["W"]=W
    cache["b"]=b 
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
    # lin_cache - {"APrev":A,"W":W,"b":b}
    Z, lin_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    
    # store  Current A as well
    lin_cache["ACur"] = A

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters)//2  
    A = X
    caches = []
    # here since multiple layers are present, we iterate using loop
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    # print cache["act_cache"]
    caches.append(cache)
    # print "hi",caches[2]["act_cache"]["Z"].shape

    return AL, caches

def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    ## CODE HERE 

    A_prev = cache["APrev"]
    W = cache["W"]
    b = cache["b"]

    # print dZ.shape,W.shape,A_prev.shape

    dA_prev = np.dot(W.T,dZ)
    dW = np.dot(dZ,A_prev.T)
    db = np.sum(dZ,axis=1,keepdims=1)

    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    # elif activation == "softmax":
    #     dZ = softmax_cross_entropy_loss_der(dA, act_cache)

    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    # print dA.shape
    # activation = "linear"
    activation="relu"
    # print type(caches)
    for l in reversed(range(1,L)):
        # print l
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE 
   
     # Forward propagate X using multi_layer_forward
    Alast,cache = multi_layer_forward(X, parameters)

    # Get predictions using softmax_cross_entropy_loss
    Z = cache[-1]["act_cache"]["Z"]
    A=[]
    n = Z.shape[1]
   
    for i in range(0,n):
        exps = np.exp(Z[:,i] - np.max(Z[:,i]))
        A.append(exps / (np.sum(exps)))
        
    A = np.array(A,dtype=np.float)
    # 10*samples
    Alast = A.T

    # Estimate the class labels using predictions
    Ypred=[]
    n = Alast.shape[1]
    mx = -1.00
    indx=1
    # print Alast[:,[101,202,303]]
    # print Y[0][101],Y[0][202],Y[0][303]
    # print type(Y[0][3])

    for j in range(0,n):
        mx = -1
        for i in range(0,10):
            if Alast[i][j] > mx:
                mx = Alast[i][j]
                indx = i
        # print Alast[:,j],Y[0][j],mx,indx
        Ypred.append(indx)


    # print type(Ypred[101])
    # print Ypred[101],Ypred[202],Ypred[303]
    Ypred  = np.array(Ypred)
    # print Ypred.shape
    return Ypred

def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.00):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''

    alpha = learning_rate*(1/(1+decay_rate*epoch))
    # print alpha
    # L = len(parameters)//2
    ### CODE HERE  DONE
    layers = len(parameters)/2
    for i in range(1,layers+1):
        parameters["W"+str(i)] = parameters["W"+str(i)] - alpha*gradients["dW"+str(i)]
        parameters["b"+str(i)] = parameters["b"+str(i)] - alpha*gradients["db"+str(i)]

    return parameters, alpha

def multi_layer_network(X, Y,VData,VLabel, net_dims, num_iterations=1000, learning_rate=0.2, decay_rate=0.00):
    '''
    Creates the multilayer network and trains the network

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
    parameters = initialize_multilayer_weights(net_dims)
    A0 = X
    # validation data
    A1 = VData
    Y1 = VLabel
    costs = []
    vcosts=[]
    cost = 999999999999
    for ii in range(num_iterations+1):
        ### CODE HERE 
        # Forward Prop

        # training data
        ## call to multi_layer_forward to get activations
        ATlast,cache = multi_layer_forward(A0, parameters)
    
        ZTlast= cache[-1]["act_cache"]["Z"]

        ## call to softmax cross entropy loss to calculate softmax for last layer and also calulating loss
        ATlast,lastActCache,cost  = softmax_cross_entropy_loss(ZTlast,Y)
    
        cache[-1]["lin_cache"]["ACur"]= ATlast

        # validation data
        ## call to multi_layer_forward to get activations
        AVlast,vcache = multi_layer_forward(A1, parameters)
    
        ZVlast= vcache[-1]["act_cache"]["Z"]

        ## call to softmax cross entropy loss to calculate softmax for last layer and also calulating loss
        AVlast,lastVActCache,vcost  = softmax_cross_entropy_loss(ZVlast,Y1)
    

        # print np.sum(A[:,i])   ---  has to be 1.. since addition of all probabilities from (0,10) indices is 1
        

        # back propagate
        # last layer derivate
        layers = len(parameters)/2
        # print layers
        dZ = softmax_cross_entropy_loss_der(Y,lastActCache)
        W = parameters["W"+str(layers)]
        APrev = cache[-1]["lin_cache"]["APrev"]
        dW = np.dot(dZ,APrev.T)
        db = np.sum(dZ,axis=1,keepdims=True)
        dAprev = np.dot(W.T,dZ)

        ## call to multi_layer_backward to get gradients of other layers
        
        gradients = multi_layer_backward(dAprev,cache,parameters)
        l = len(cache)
        # print l
        # since other layers are independent of this, it desnt matter even if we assign after completion of back propagation
        gradients["dW"+str(l)] = dW
        gradients["db"+str(l)] = db

        ## call to update the parameters
        parameteres, alpha = update_parameters(parameters, gradients, num_iterations, learning_rate, decay_rate=0.00)

        
        if ii % 10 == 0:
            costs.append(cost)
            vcosts.append(vcost)
        if ii % 1000 == 0:
            print("Training Cost at iteration %i is: %.05f " %(ii, cost))
            print("Validation Cost at iteration %i is: %.05f" %(ii, vcost))
                    
    
    return costs,vcosts,parameters

def calculateAccuracy(Y,YPred):
    n = Y.shape[1]
    count=0
    y = Y.tolist()[0]
    # print YPred.shape,Y.shape

    for i in range(0,n):
        # print Y[0][i],YPred[i]
        if(int(y[i])==int(YPred[i])):
            count+=1
    accuracy = (float(count)/float(n))*100
    print accuracy
    return accuracy


def main():
    '''
    Trains a multilayer network for MNIST digit classification (all 10 digits)
    To create a network with 1 hidden layer of dimensions 800
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800]"
    The network will have the dimensions [784,800,10]
    784 is the input size of digit images (28pix x 2    8pix = 784)
    10 is the number of digits

    To create a network with 2 hidden layers of dimensions 800 and 500
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800,500]"
    The network will have the dimensions [784,800,500,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits
    '''

    # getting the subset dataset from MNIST
    data, label, test_data, test_label = \
            mnist(noTrSamples=6000,noTsSamples=1000,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=600, noTsPerClass=100)
   
    # parse data into training and validation sets
    # training data
    train_data=np.array([])
    train_label=np.array([])
    train_data= data[:,0:500]
    train_label1 =label[:,0:500]
    for i in range(600,6000,600):
        # print i
        temp = data[:,i:i+500]
        temp2 = label[:,i:i+500]
        # print temp.shape
        train_data=np.column_stack((train_data,temp))
        train_label1 = np.column_stack((train_label1,temp2))

    # train data -> 784*5000
    # reshape train_lable from 1*5000 to 10*5000
    # since instead of train_label[sample]=digit it should be train_label[sample][digit] = 1/0
    column = train_label1.shape[1]
    mainList=[]
    for i in range(0,column):
        l=[0,0,0,0,0,0,0,0,0,0]        
        temp = int(train_label1[0][i])
        l[temp]=1
        mainList.append(l)

    train_label = np.array(mainList)
    train_label = train_label.T
    
    # print "Train labels",train_label.shape
   
    # validation data
    validation_data=np.array([])
    validation_label=np.array([])
    validation_data= data[:,500:600]
    validation_label1 =label[:,500:600]
    for i in range(1100,6000,600):
        # print i
        temp = data[:,i:i+100]
        temp2 = label[:,i:i+100]
        # print temp.shape
        validation_data=np.column_stack((validation_data,temp))
        validation_label1 = np.column_stack((validation_label1,temp2))

    # train data -> 784*5000
    # reshape train_lable from 1*5000 to 10*5000
    # since instead of train_label[sample]=digit it should be train_label[sample][digit] = 1/0
    
    column = validation_label1.shape[1]
    mainList=[]
    for i in range(0,column):
        l=[0,0,0,0,0,0,0,0,0,0]        
        temp = int(validation_label1[0][i])
        l[temp]=1
        mainList.append(l)

    validation_label = np.array(mainList)
    validation_label = validation_label.T


    # validation data -> 784*1000

    # confirm data 
    # a =[0,0,0,0,0,0,0,0,0,0]
    # for i in validation_label[0]:
    #     a[int(i)]+=1
    # print a

     # initialize learning rate and num_iterations

     # varying learning rates.
    num_iterations = 1000
    rates = [0.1, 0.2, 0.5, 1.0, 10]
    net_dims = ast.literal_eval( sys.argv[1] )
        # print net_dims
    net_dims.append(10) # Adding the digits layer with dimensionality = 10
    print "Network dimensions are:" + str(net_dims) 

    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
    l = [ax1,ax2,ax3,ax4,ax5]
    j=0
    for learning_rate in rates:
        print "Learning Rate: ", learning_rate
       
        costs, vcosts,parameters = multi_layer_network(train_data, train_label,validation_data,validation_label, net_dims, \
                num_iterations=num_iterations, learning_rate=learning_rate)
        
        # compute the accuracy for training set and testing set
        train_Pred = classify(train_data, parameters)
        test_Pred = classify(test_data, parameters)

        validation_Pred = classify(validation_data , parameters)

        trAcc = calculateAccuracy(train_label1,train_Pred)
        teAcc = calculateAccuracy(test_label,test_Pred)
        vaAcc = calculateAccuracy(validation_label1,validation_Pred)

        print "Accuracy for training set is {0:0.3f} %".format(trAcc)
        print "Accuracy for Validation set is {0:0.3f} %".format(vaAcc)
        print "Accuracy for testing set is {0:0.3f} %".format(teAcc)
        
        
        ### CODE HERE to plot costs
        iterations=[i for i in range(0,1001,10)]
       
        # ,iterations,vcosts,'b',label='Validation cost'
        
        l[j].plot(iterations,costs,'r',label='Training cost')
        l[j].plot(iterations,vcosts,'b',label='Validation cost')
        title = "Learning rate: ", learning_rate
        l[j].set_title(title)
        l[j].set_xlabel('Iterations')
        l[j].set_ylabel('Costs')
        l[j].legend()
        j+=1

    plt.show()


if __name__ == "__main__":
    main()