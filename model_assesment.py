__author__ = 'dipanjan'
import numpy as np
import scipy as sc
print np.__version__
print sc.__version__
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import data_generator_learning as dgl
import data_generator_validation as dgv
import data_generator_test as dgt
from six.moves import cPickle as pickle
import model_generator as modelgn


class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 1  # output layer dimensionality
    # Gradient descent parameters (hard coded)
    epsilon = 0.0001  # learning rate for gradient descent
    reg_lambda = 0.0000000001  # regularization strength
    sample_dim =100   # sample size
    sample_dim_valid =30   # sample size
    sample_dim_test =10  # sample size


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







def visualizePredictedCurveUsingLinearRegression(X80,y80,X20,y20):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X80, y80)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    
    # The mean square error
    print('Residual sum of squares: %.2f' % np.mean((regr.predict(X20) - y20) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X20, y20))
    output = regr.predict(X20)
    plt.subplot(3, 1, 3)
    #plt.title('Hidden Layer size %d' % nn_hdim)
    plt.title("LinearRegression Generated PDF")
    colors = np.random.rand(Config.sample_dim)
    area = np.pi * (5 * np.random.rand(Config.sample_dim))**2
    plt.scatter(X20[:,0],output, s=area, c=colors, alpha=0.5)


def visualizePredictedCurve(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    #print model
    output =predict(model, X)
    
    #plot_decision_boundary(lambda x:predict(model,x), X, y)
    #print output
    #print X[:,0]
    plt.subplot(3, 1, 2)
    #plt.title('Hidden Layer size %d' % nn_hdim)
    plt.title("Deep Learning Generated PDF")
    colors = np.random.rand(Config.sample_dim)
    area = np.pi * (5 * np.random.rand(Config.sample_dim))**2
    plt.scatter(X[:,0],output, s=area, c=colors, alpha=0.5)
#plt.show()

def visualizeActualCurve(X, y):
    plt.subplot(3, 1, 1)
    #plt.title('Hidden Layer size %d' % nn_hdim)
    plt.title("Actual PDF")
    colors = np.random.rand(Config.sample_dim)
    area = np.pi * (5 * np.random.rand(Config.sample_dim))**2
    plt.scatter(X[:,0],y, s=area, c=colors, alpha=0.5)
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


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    #exp_scores = np.exp(z2)
    #probs = exp_scores
    return z2



def classify(X, y):
    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X, y)
    # return clf

    pass


def main():
    #generate_data()
    data ,y ,data_valid,y_valid,data_test,y_test = modelgn.load_data()
    np.seterr( over='ignore' )
    visualizeActualCurve(data,y)
    model=modelgn.load_model('model.pickle')
    visualizePredictedCurve(data_test, y_test, model)
    print("Test Loss  %f" % ( calculate_loss(model, data_test, y_test)))
    visualizePredictedCurveUsingLinearRegression(data, y,data_test,y_test)
    plt.show()


if __name__ == "__main__":
    main()
