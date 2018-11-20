'''
Generates two moon dataset

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import sklearn.metrics.pairwise

from sklearn import svm


def genTwoMoons(n=400, radius=1.5, width=0.5, dist=-1):
    rho = radius-width/2 + width*np.random.rand(1,n)
    phi = np.pi*np.random.randn(1,n)
    X = np.zeros((2,n))
    X[0], X[1] = polar2cart(rho, phi)
    id = X[1]<0
    X[0,id] = X[0,id] + radius
    X[1,id] = X[1,id] - dist
    Y = np.zeros(n)
    Y[id] = 1
    return X, Y


def polar2cart(rho, phi):
    x =  rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y



def svmFun(X,Y,CValue,gammaValue,kernel):
    # print type(Y[0]),X
    clf = svm.SVC(C=CValue, gamma=gammaValue, kernel=kernel, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None)
    clf.fit(X.T, Y)

    return clf


def plot_contours(xx, yy,Z):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z,)
    return out



def accuracy(clf,X,Y):
    X = X.T
    Z =clf.predict(X)
    n = X.shape[0]
    count=0
    for i in range(n):
        if(Z[i]==Y[i]):
            count+=1
    acc = float((float(count)/float(n))*100)
    # print accuracy
    return "{0:.3f}".format(round(acc,2))  




def main():
    XTr, YTr = genTwoMoons(n=400, radius=1.5, width=0.5, dist=-1)
    XVa,YVa = genTwoMoons(n=100, radius=1.5, width=0.5, dist=-1)
    XTs,YTs = genTwoMoons(n=200, radius=1.5, width=0.5, dist=-1)
    colors = ['red','blue']


    C =[0.001, 0.1, 10, 1000]
    gamma = [0.001, 0.01, 0.1]

    print "\nrbf Kernel:"
    for c in C:
        for gam in gamma:
            print "Gamma:", gam,"C:", c
            clf = svmFun(XTr,YTr,c,gam,"rbf")

            VaAcc =  accuracy(clf,XVa,YVa)
            print "Validation accuracy is: ", VaAcc


    print "\nPolynomial kernel:"
    for c in C:
        for gam in gamma:
            print "Gamma: ", gam,"C: ", c
            clf = svmFun(XTr,YTr,c,gam,"poly")

            VaAcc =  accuracy(clf,XVa,YVa)
            print "Validation accuracy is: ", VaAcc



    print "Best validation Accuracy for both kernels is when C=1000 and gamma=0.1"
    
    clf = svmFun(XTr,YTr,1000,0.1,"rbf")
    TsAcc =  accuracy(clf,XTs,YTs)    
    print "Test accuracy for C = 1000 and gamma = 0.1 for rbf kernel: ", TsAcc

    clf = svmFun(XTr,YTr,1000,0.1,"poly")
    TsAcc =  accuracy(clf,XTs,YTs)
    print "Test accuracy for C = 1000 and gamma = 0.1 for polynomial kernel: ",TsAcc 



# plot_contours

    # Plotting decision regions
    # print XTr.shape

    C =[1000,10]
    gamma =[0.1,0.01]

    # plot for different C and gamma
    
    fig, axs = plt.subplots(2, 2)
    plt.tight_layout(pad=3.5, w_pad=3.5, h_pad=3.5)
    fig.suptitle("\nDecision Boundaries\n")
    for i in range(0,2):
        for j in range(0,2):

            # classify using training
            c = C[i]
            gam = gamma[j]
            clf = svmFun(XTr,YTr,c,gam,"rbf")

            # concatenate train and test data 
            X = np.column_stack((XTr,XTs))
            Y = np.append(YTr,YTs)

            # Accuracy on X and Y as complete dataset
            Acc =  accuracy(clf,X,Y)  

            # X = XTr
            # Y= YTr
            x_min, x_max = X[0].min() - 1, X[0].max() + 1
            y_min, y_max = X[1].min() - 1, X[1].max() + 1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            

            axs[i][j].contourf(xx, yy, Z,alpha=0.6)
            axs[i][j].scatter(X[0], X[1],c=Y, s=20, edgecolor='k')
            
            title = "C: "+ str(c) +"\nGamma: " + str(gam) + "\nAccuracy: "+ str(Acc)
            axs[i][j].set_title(title)
            

    plt.show()

  

if __name__ == "__main__":
    main()


    # TrAcc = accuracy(clf,XTr,YTr)
    # print "Training accuracy is: ", TrAcc


    # TsAcc =  accuracy(clf,XTs,YTs)
    # print "Test accuracy is: ", TsAcc

  # fig = plt.figure(figsize=(6,4))
    # plt.scatter(XTr[0], XTr[1], c=YTr, cmap=clr.ListedColormap(colors))
    # plt.show()
    
    # plt.scatter(XVa[0], XVa[1], c=YVa, cmap=clr.ListedColormap(colors))
    # plt.show()

    # plt.scatter(XTs[0], XTs[1], c=YTs, cmap=clr.ListedColormap(colors))
    # plt.show()