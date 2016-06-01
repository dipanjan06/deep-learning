__author__ = 'dipanjan'
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

version = '0.1'
class Config:
    
    sample_dim =1000  # sample size
    b=0.5
    w=0.3
    startpoint=1
    endpoint=100



def generate_and_store_data(dim):
    #np.random.seed(0)
    #X = np.random.randint(Config.startpoint,Config.endpoint,size=dim) # discrete uniform distribution
    #X = np.random.uniform(-10,10,size=(dim))
    
    X = np.random.normal(0, 0.4, dim)

    
    #y = 0.1*X+0.5*np.power(X,2)
    #+0.2*np.power(X,3)+0.1*np.power(X,4)+0.01*np.power(X,5)
    y=Config.b+Config.w*X
    
    np.savetxt(
               'RawData.csv',           # file name
               X,                # array to save
               fmt='%.10f',             # formatting, 2 digits in this case
               delimiter=',',          # column delimiter
               newline='\n',           # new line character
               footer='end of file',   # file footer
               comments='# ',          # character to use for comments
               header='Input Data')      # file header
    np.savetxt(
                'LearningSet.csv',           # file name
                y,                # array to save
                fmt='%.10f',             # formatting, 10 digits in this case
                delimiter=',',          # column delimiter
                newline='\n',           # new line character
                footer='end of file',   # file footer
                comments='# ',          # character to use for comments
                header='Output Data')      # file header

    colors = np.random.rand(Config.sample_dim)
    area = np.pi * (3 * np.random.rand(Config.sample_dim))**2
    plt.scatter(X,y, s=area, c=colors, alpha=0.5)
    plt.show()

def read_data():
    X = np.genfromtxt(
                            'RawData.csv',           # file name
                            skip_header=0,          # lines to skip at the top
                            skip_footer=0,          # lines to skip at the bottom
                            delimiter=',',          # column delimiter
                            dtype='f8',        # data type
                            filling_values=0,       # fill missing values with 0
                            usecols = (0),    # columns to read
                            names=['first'])     # column names

    Y = np.genfromtxt(
                  'LearningSet.csv',           # file name
                  skip_header=0,          # lines to skip at the top
                  skip_footer=0,          # lines to skip at the bottom
                  delimiter=',',          # column delimiter
                  dtype='float32',        # data type
                  filling_values=0,       # fill missing values with 0
                  usecols = (0),    # columns to read
                  names=['first'])     # column names
#    print(X)
#    print(X.shape)
#    X =np.array(X).astype('float')
#    print('*******************')
#    print(X)
#    print('*******************')
#    print(X.shape)
#    X= X*1
    return X ,Y

#    colors = np.random.rand(Config.sample_dim)
#    area = np.pi * (3 * np.random.rand(Config.sample_dim))**2
#    plt.scatter(X,y, s=area, c=colors, alpha=0.5)
#    plt.show()


def main():
    generate_and_store_data(Config.sample_dim)

    read_data()


if __name__ == "__main__":
    main()
