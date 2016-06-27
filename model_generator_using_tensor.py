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
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import math

num_steps_for_batch=10001
batch_size = 128
hidden1_units=800
hidden2_units=800
regularizers_constant =0.0000000000001

class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 1  # output layer dimensionality
    # Gradient descent parameters (hard coded)
    #epsilon = 0.0001  # learning rate for gradient descent
    #reg_lambda = 0.0000000001  # regularization strength
    sample_dim = 1024   # sample size
    sample_dim_valid = 256   # sample size
    sample_dim_test = 1024  # sample size
    x_dim = 2
    num_labels = 1




# This function generate learning, validation and test data set
def generate_data():
    dgl.generate_and_store_data(Config.sample_dim)
    dgv.generate_and_store_data(Config.sample_dim_valid)
    dgt.generate_and_store_data(Config.sample_dim_test)

# This function load learning, validation and test data sets into nump array and append zeros as second dimenson
def load_data():

    
    data , y= dgl.read_data()
    data = np.array(data).astype('float32')
    y = np.array(y).astype('float32')
    dummy=np.zeros((Config.sample_dim,),dtype=np.float32)
    data=np.reshape(data, (Config.sample_dim, 1) )
    dummy=np.reshape(dummy, (Config.sample_dim, 1) )
    data = np.append(data, dummy,axis=1)
    y=np.reshape(y, (Config.sample_dim, 1) )
    
    data_valid , y_valid= dgv.read_data()
    data_valid = np.array(data_valid).astype('float32')
    y_valid = np.array(y_valid).astype('float32')
    dummy_valid=np.zeros((Config.sample_dim_valid,),dtype=np.float32)
    dummy_valid=np.reshape(dummy_valid, (Config.sample_dim_valid, 1) )
    data_valid=np.reshape(data_valid, (Config.sample_dim_valid, 1) )
    data_valid = np.append(data_valid, dummy_valid,axis=1)
    y_valid=np.reshape(y_valid, (Config.sample_dim_valid, 1) )
    
    data_test , y_test= dgt.read_data()
    data_test = np.array(data_test).astype('float32')
    y_test = np.array(y_test).astype('float32')
    dummy_test=np.zeros((Config.sample_dim_test,),dtype=np.float32)
    dummy_test=np.reshape(dummy_test, (Config.sample_dim_test, 1) )
    data_test=np.reshape(data_test, (Config.sample_dim_test, 1) )
    data_test = np.append(data_test, dummy_test,axis=1)
    y_test=np.reshape(y_test, (Config.sample_dim_test, 1) )
    
    print('********')
    print(data.shape)
              
    return data ,y ,data_valid,y_valid,data_test,y_test







def inference_for_network(X,tf_test_dataset, hidden1_units, hidden2_units,isTest,nlf='relu'):
    # Hidden 1
    with tf.name_scope('hidden1'):

        weights = tf.Variable(tf.truncated_normal([Config.x_dim, hidden1_units],stddev=1.0 / math.sqrt(float(Config.x_dim))),name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
        
        
        malproduct = tf.cond(isTest, lambda: (tf.matmul(tf_test_dataset, weights) + biases), lambda: (tf.matmul(X, weights) + biases))
        #malproduct = tf.matmul(X, weights) + biases
#        if(tf.equal(isTest, [1.0]).True):
#            malproduct = tf.matmul(tf_test_dataset, weights) + biases
#        else:
#            malproduct = tf.matmul(X, weights) + biases

        if(nlf == 'sigmoid'):
            hidden1 = tf.nn.sigmoid(malproduct)
        elif(nlf == 'relu') :
            hidden1 = tf.nn.relu(malproduct)
        else :
            hidden1 = tf.nn.tanh(malproduct)
                                 
        regloss1 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
        if(nlf == 'sigmoid'):
            hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights) + biases)
        elif(nlf == 'relu'):
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        else:
            hidden2 = tf.nn.tanh(tf.matmul(hidden1, weights) + biases)
        
        regloss2 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
    # Linear
    with tf.name_scope('reg_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, Config.num_labels],stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
        biases = tf.Variable(tf.zeros([Config.num_labels]),name='biases')
        activation = tf.matmul(hidden2, weights) + biases
        regloss3 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
    return activation ,(regloss1+regloss2+regloss3)





def loss(activation, y ,regularizers):
    #labels = tf.to_int64(labels)
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(activation, y)
    #loss = tf.reduce_mean(cross_entropy,name='CrossEntLoss')
    
    loss = tf.reduce_sum(((activation-y)**2)/batch_size) #Data  loss
    #loss = (activation-y)*0.5
    #loss = tf.reduce_sum(tf.square(activation-y))#/(2*batch_size) #Data  loss
    #loss = tf.sub(activation ,y)
    
    loss += regularizers_constant * regularizers
    
    return loss
                         
def training(loss, learning_rate):
    """Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
        Returns:
        train_op: The Op for training.
        """
    # Add a scalar summary for the snapshot loss.
    #tf.scalar_summary(loss.op.name, loss)
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op



def training_with_momentum(loss,train_size):
    tf.scalar_summary(loss.op.name, loss)
    batch = tf.Variable(0,name='DecayBatch')
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(0.01,
                                               batch * batch_size,  # Current index into the dataset.
                                               train_size,          # Decay step.
                                               0.95,                # Decay rate.
                                               staircase=True)
                                               # Use simple momentum for the optimization.
    train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)
    
    
    return train_op

def traing_with_admoptimizer(loss,train_size):
    tf.scalar_summary(loss.op.name, loss)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return train_op;

def run_training(train_dataset, train_labels,valid_dataset, valid_labels,test_dataset,test_labels,nlf):

    
    
    print("batch_gradient_descen")
    
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, Config.x_dim),name='TraingSet')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, Config.num_labels),name='TraingLabel')
        tf_valid_dataset = tf.placeholder(tf.float32,shape=(batch_size, Config.x_dim),name='ValidationSet')
        tf_valid_labels = tf.placeholder(tf.float32, shape=(batch_size, Config.num_labels),name='ValidationLabel')
        tf_test_dataset = tf.constant(test_dataset)
        # Build a Graph that computes predictions from the inference model.
        
        isTest = tf.placeholder(tf.bool)
        
        activation ,regloss = inference_for_network(tf_train_dataset,tf_test_dataset,hidden1_units,hidden2_units,isTest,nlf)
        
        # Add to the Graph the Ops for loss calculation.
        loss_tr = loss(activation, tf_train_labels,regloss)
        
        # Add to the Graph the Ops that calculate and apply gradints.
        #train_op = training(loss_tr, 0.5)
        #train_op = training(loss_tr, 0.5)
        #train_op = training_with_momentum(loss_tr,Config.sample_dim)
        train_op = traing_with_admoptimizer(loss_tr,Config.sample_dim)
        
        #tp = prediction(activation,'traing_softmax_for_predict')
        #vp = prediction(activation,'validation_softmax_for_predict')
        
        #        logits_validation = inference_for_network(tf_valid_dataset,hidden1_units,hidden2_units)
        #
        #valid_prediction=prediction(logits)
        #
        #        logits_test = inference_for_network(tf_test_dataset,hidden1_units,hidden2_units)
        #
        #testp=prediction(logits,'test_softmax_for_predict')
        
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        summary_writer = tf.train.SummaryWriter('TrainingDir', session.graph)
        for step in range(num_steps_for_batch):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,isTest :False}
            _, l, predictions = session.run([train_op, loss_tr, activation], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                summary_str = session.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            offset = (step * batch_size) % (valid_labels.shape[0] - batch_size)
            batch_data = valid_dataset[offset:(offset + batch_size), :]
            batch_labels = valid_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,isTest : False}
            val_loss = session.run(loss_tr, feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Validation loss: %f" % val_loss)
        saver.save(session,'model.ckpt')


        batch_data = test_dataset[0:(0 + batch_size), :]
        batch_labels = test_labels[0:(0 + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,isTest : True }
        activation = session.run(activation, feed_dict=feed_dict)

        plt.subplot(3, 1, 1)
            #plt.title('Hidden Layer size %d' % nn_hdim)
        plt.title("Actual PDF")
        colors = np.random.rand(Config.sample_dim)
        area = np.pi * (5 * np.random.rand(Config.sample_dim))**2
        plt.scatter(train_dataset[:,0],train_labels, s=area, c=colors, alpha=0.5)

        plt.subplot(3, 1, 2)
        plt.title("Deep Learning Generated PDF")
        colors = np.random.rand(Config.sample_dim_test)
        area = np.pi * (5 * np.random.rand(Config.sample_dim_test))**2
        plt.scatter(test_dataset[:,0],activation, s=area, c=colors, alpha=0.5)



        # Create linear regression object
        regr = linear_model.LinearRegression()
    
        # Train the model using the training sets
        regr.fit(train_dataset, train_labels)
    
        # The coefficients
        print('Coefficients: \n', regr.coef_)
    
        # The mean square error
        print('Residual sum of squares: %.2f' % np.mean((regr.predict(test_dataset) - test_labels) ** 2))
            # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % regr.score(test_dataset, test_labels))
        output = regr.predict(test_dataset)
        plt.subplot(3, 1, 3)
    
        plt.title("LinearRegression Generated PDF")
        colors = np.random.rand(Config.sample_dim_test)
        area = np.pi * (5 * np.random.rand(Config.sample_dim_test))**2
        plt.scatter(test_dataset[:,0],output, s=area, c=colors, alpha=0.5)
        plt.show()



    

#return model
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
def accuracy(predictions, labels):
    return (100.0 * (predictions - labels)/predictions.shape[0])

def main():
    #generate_data()
    train_dataset, train_labels,valid_dataset, valid_labels,test_dataset,test_labels = load_data()
    np.seterr( over='ignore' )
    #visualizeActualCurve(data,y)
    run_training(train_dataset, train_labels,valid_dataset, valid_labels,test_dataset,test_labels,'relu')
    #model = build_model(data, y, 5,10000,data_valid,y_valid, print_loss=True)
#store_model(model,'model.pickle')
    #visualizePredictedCurve(data_test, y_test, model)
    #print("Test Loss  %f" % ( calculate_loss(model, data_test, y_test)))
    #visualizePredictedCurveUsingLinearRegression(data, y,data_test,y_test)
#plt.show()


if __name__ == "__main__":
    main()
