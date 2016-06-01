__author__ = 'dipanjan'

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import data_generator_learning as dgl
import data_generator_validation as dgv
import data_generator_test as dgt

from six.moves import cPickle as pickle
from sklearn import datasets, linear_model


class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 1  # output layer dimensionality
    # Gradient descent parameters (hard coded)
    epsilon = 0.0001  # learning rate for gradient descent
    reg_lambda = 0.0000000001  # regularization strength
    sample_dim =1000   # sample size
    sample_dim_valid =300   # sample size
    sample_dim_test =1000  # sample size


#def read_data(filename):
#    X = np.genfromtxt(
#                      filename,           # file name
#                      skip_header=0,          # lines to skip at the top
#                      skip_footer=0,          # lines to skip at the bottom
#                      delimiter=',',          # column delimiter
#                      dtype='float32',        # data type
#                      filling_values=0,       # fill missing values with 0
#                      usecols = (0),    # columns to read
#                      names=['first'])     # column namesur
#
## print('Shape of '+filename)
#    np.shape(X)
#    return X


# This function generate learning, validation and test data set
def generate_data():
    dgl.generate_and_store_data(Config.sample_dim)
    dgv.generate_and_store_data(Config.sample_dim_valid)
    dgt.generate_and_store_data(Config.sample_dim_test)

# This function load learning, validation and test data sets into nump array and append zeros as second dimenson
def load_data():

    
    data , y= dgl.read_data()
    data = np.array(data).astype('float')
    y = np.array(y).astype('float')
    dummy=np.zeros((Config.sample_dim,),dtype=np.float32)
    data=np.reshape(data, (Config.sample_dim, 1) )
    dummy=np.reshape(dummy, (Config.sample_dim, 1) )
    data = np.append(data, dummy,axis=1)
    y=np.reshape(y, (Config.sample_dim, 1) )
    
    data_valid , y_valid= dgv.read_data()
    data_valid = np.array(data_valid).astype('float')
    y_valid = np.array(y_valid).astype('float')
    dummy_valid=np.zeros((Config.sample_dim_valid,),dtype=np.float32)
    dummy_valid=np.reshape(dummy_valid, (Config.sample_dim_valid, 1) )
    data_valid=np.reshape(data_valid, (Config.sample_dim_valid, 1) )
    data_valid = np.append(data_valid, dummy_valid,axis=1)
    y_valid=np.reshape(y_valid, (Config.sample_dim_valid, 1) )
    
    data_test , y_test= dgt.read_data()
    data_test = np.array(data_test).astype('float')
    y_test = np.array(y_test).astype('float')
    dummy_test=np.zeros((Config.sample_dim_test,),dtype=np.float32)
    dummy_test=np.reshape(dummy_test, (Config.sample_dim_test, 1) )
    data_test=np.reshape(data_test, (Config.sample_dim_test, 1) )
    data_test = np.append(data_test, dummy_test,axis=1)
    y_test=np.reshape(y_test, (Config.sample_dim_test, 1) )
    
              
    return data ,y ,data_valid,y_valid,data_test,y_test

#def visualizePredictedCurveUsingLinearRegression(X80,y80,X20,y20):
#    # Create linear regression object
#    regr = linear_model.LinearRegression()
#
#    # Train the model using the training sets
#    regr.fit(X80, y80)
#
#    # The coefficients
#    print('Coefficients: \n', regr.coef_)
#    
#    # The mean square error
#    print('Residual sum of squares: %.2f' % np.mean((regr.predict(X20) - y20) ** 2))
#    # Explained variance score: 1 is perfect prediction
#    print('Variance score: %.2f' % regr.score(X20, y20))
#    output = regr.predict(X20)
#    plt.subplot(3, 1, 3)
#    #plt.title('Hidden Layer size %d' % nn_hdim)
#    plt.title("LinearRegression Generated PDF")
#    colors = np.random.rand(Config.sample_dim)
#    area = np.pi * (5 * np.random.rand(Config.sample_dim))**2
#    plt.scatter(X20[:,0],output, s=area, c=colors, alpha=0.5)


#def visualizePredictedCurve(X, y, model):
#    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
#    # plt.show()
#    #print model
#    output =predict(model, X)
#    
#    #plot_decision_boundary(lambda x:predict(model,x), X, y)
#    #print output
#    #print X[:,0]
#    plt.subplot(3, 1, 2)
#    #plt.title('Hidden Layer size %d' % nn_hdim)
#    plt.title("Deep Learning Generated PDF")
#    colors = np.random.rand(Config.sample_dim)
#    area = np.pi * (5 * np.random.rand(Config.sample_dim))**2
#    plt.scatter(X[:,0],output, s=area, c=colors, alpha=0.5)
#plt.show()

#def visualizeActualCurve(X, y):
#    plt.subplot(3, 1, 1)
#    #plt.title('Hidden Layer size %d' % nn_hdim)
#    plt.title("Actual PDF")
#    colors = np.random.rand(Config.sample_dim)
#    area = np.pi * (5 * np.random.rand(Config.sample_dim))**2
#    plt.scatter(X[:,0],y, s=area, c=colors, alpha=0.5)
#plt.show()




# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    num_examples = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    data_loss =  np.mean((z2 - y) ** 2)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


#def predict(model, x):
#    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
#    # Forward propagation
#    z1 = x.dot(W1) + b1
#    a1 = np.tanh(z1)
#    z2 = a1.dot(W2) + b2
#    #exp_scores = np.exp(z2)
#    #probs = exp_scores
#    return z2


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes,data_valid,y_valid, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)
    #print num_examples
    #np.random.seed(0)
    print(nn_hdim)
    
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))
    #print np.shape(W1)
    #print W1
    #print "Dimension of W2"
    #print np.shape(W2)
    #print W2
    # This is what we return at the end
    model = {}
    
    dd=np.array([2,3])
    yy=np.array([1])
    
    #dd[range(1), yy] -= 1
    #print yy
    
    
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
        
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        #exp_scores = np.exp(z2)
#        print('z2___________')
#        print(z2)
#        print('z2______________')


        # Backpropagation
        delta3 = z2
        #print "prob"
        #print delta3
        #print "range(num_examples)"
        #print range(num_examples)
        delta3=(delta3-y)*.5#[range(num_examples), y] -= 1
        #tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples)
        #delta3=np.sum(np.power(delta3-y, 2))/(2*num_examples)
        #print "Error"
        #print delta3
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2*W2
        dW1 += Config.reg_lambda * W1*W1
        
        # Gradient descent parameter update
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2
        
        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 100 == 0:
           print("Training Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))
           print("Validation Loss after iteration %i: %f" % (i, calculate_loss(model, data_valid, y_valid)))

#    model_json ={"W1":[W1],"W2":[W2],"b1":[b1],"b2":[b2]}
#    with open('model.txt', 'w') as txtfile:
#        json.dump(model, txtfile)
#    print(model)
#    try:
#        with open('model.pickle', 'wb') as f:
#            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
#    except Exception as e:
#                print('Unable to save data to', 'model.pickle', ':', e)
#    try:
#        with open('model.pickle', 'rb') as f:
#            model1 = pickle.load( f)
#            print(model1)
#    except Exception as e:
#        print('Unable to load', 'model.pickle', ':', e)
    return model
#This function store the generated model into file
def store_model(model,filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', filename, ':', e)

#This function load the  model from file
def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load( f)
    except Exception as e:
        print('Unable to load', filename, ':', e)
    return model
def classify(X, y):
    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X, y)
    # return clf

    pass


def main():
    generate_data()
    data ,y ,data_valid,y_valid,data_test,y_test = load_data()
    np.seterr( over='ignore' )
    #visualizeActualCurve(data,y)
    model = build_model(data, y, 200,10000,data_valid,y_valid, print_loss=True)
    store_model(model,'model.pickle')
    #visualizePredictedCurve(data_test, y_test, model)
    #print("Test Loss  %f" % ( calculate_loss(model, data_test, y_test)))
    #visualizePredictedCurveUsingLinearRegression(data, y,data_test,y_test)
#plt.show()


if __name__ == "__main__":
    main()
