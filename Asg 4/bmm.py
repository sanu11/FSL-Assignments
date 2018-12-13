import numpy as np

def bmm(data):

    # initialise thetha
    theta_A =  np.random.random()
    theta_B =  np.random.random()
    thetas = [(theta_A, theta_B)]
 
    iterations=20
    for i in range(iterations):
        print theta_A, theta_B
        heads_A, tails_A, heads_B, tails_B = expectation(data, theta_A, theta_B)
        theta_A, theta_B = maximization(heads_A, tails_A, heads_B, tails_B)
        
    thetas.append((theta_A,theta_B)) 
    print "\nFinal thetas"
    print theta_A,theta_B   
    return thetas

def coin_likelihood(data, bias):

    numHeads = data.count("H")
    flips = len(data)
    return pow(bias, numHeads) * pow(1-bias, flips-numHeads)

def expectation(data, theta_A, theta_B):
    
    heads_A=0
    tails_A=0
    heads_B=0
    tails_B =0

    for trial in data:
        
        likelihood_A = coin_likelihood(trial, theta_A)
        likelihood_B = coin_likelihood(trial, theta_B)
        
        p_A = likelihood_A / (likelihood_A + likelihood_B)
        p_B = likelihood_B / (likelihood_A + likelihood_B)

        heads_A += p_A * trial.count("H")
        tails_A += p_A * trial.count("T")
        heads_B += p_B * trial.count("H")
        tails_B += p_B * trial.count("T") 

    return heads_A, tails_A, heads_B, tails_B


def maximization(heads_A, tails_A, heads_B, tails_B):

    theta_A = heads_A / (heads_A + tails_A)
    theta_B = heads_B / (heads_B + tails_B)
    
    return theta_A, theta_B


if __name__ == '__main__':
    n=1000
    m=100
    data1 = []
    with open('flips.txt', 'r') as file:
        i=0
        for row in file:
            l= row.split()
            data1.append(l)
            i+=1
    # data = np.array(data)
    # print data
    data=[]
    for trial in data1:
        l=[]
        for toss in trial:
            if toss == '1':
                l.append('H')
            else:
                l.append('T')
        data.append(l)
    thetas = bmm(data)
    # print thetas[0]
