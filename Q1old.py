import numpy as np
from numpy.core.fromnumeric import size
from numpy.core.numeric import indices

class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        
        self.w={}

        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        if activation == 'relu':
            self.act_fn = self.relu
            self.act_grad = self.relu_grad
        if activation == 'sigmoid':
            self.act_fn = self.sigmoid
            self.act_grad = self.sigmoid_grad
        if activation == 'linear':
            self.act_fn = self.linear
            self.act_grad = self.linear_grad
        if activation == 'tanh':
            self.act_fn = self.tanh
            self.act_grad = self.tanh_grad
        if activation == 'softmax':
            self.act_fn = self.softmax
            self.act_grad = self.softmax_grad

        for i in range(1,n_layers):
            if weight_init=='random':
                self.w[i] = self.random_init((layer_sizes[i], 1+layer_sizes[i-1]))    # one row represents one neuron, each neuron with w1...wn as n columns and b as 1 bias
                # self.b[i] = np.random.rand(layer_sizes[i], 1)
            elif weight_init=='zero':
                self.w[i] = self.zero_init((layer_sizes[i], 1+layer_sizes[i-1]))    # one row represents one neuron, each neuron with w0...wn as n+1 columns
                # self.b[i] = np.zeros(layer_sizes[i], 1)
            else:   # normal
                self.w[i] = self.normal_init((layer_sizes[i], 1+layer_sizes[i-1]))    # one row represents one neuron, each neuron with w0...wn as n+1 columns
                # self.b[i] = np.random.normal(size=(layer_sizes[i], 1))



    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return (X>0)*X

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return (X>0)*1

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1.0 / (1 + np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        sig = self.sigmoid(X)
        return sig * (1-sig)

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        eX = np.exp(X)
        e_X = np.exp(-X)
        return (eX - e_X) / (eX + e_X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1 - np.power(self.tanh(X), 2)

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        exps=np.exp(X-X.max())
        return exps/np.sum(exps)

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        soft = self.softmax(X)
        return soft * (1-soft)



    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeros(shape)

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return 0.01*np.random.rand(shape[0],shape[1])

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        np.random.seed(1234)
        return 0.01*np.random.randn(shape[0],shape[1])



    def CE(self, y_, ao):
        ao = ao.reshape(-1)
        y_ = y_.reshape(-1)
        
        err = -np.sum(y_*np.log(ao) + (1-y_)*np.log(1-ao))

        return err


    def CE_batch(self, y_, ypred):
        return -np.sum( y_*np.log(ypred) + (1-y_)*np.log(1-ypred) )/y_.shape[0]



    def fit(self, X, y, Xtest=None, ytest=None, save_error=False):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        # X is n(samples) x m(features)
        # y is n(samples)
        # y_ is n(samples) x c(classes)

        if save_error:
            self.train_CE=[]
            self.test_CE=[]

        y_ = np.eye(y.max()+1)[y]
        ytest_ = None
        if ytest is not None:
            ytest_ = np.eye(ytest.max()+1)[ytest]

        for epoch in range(self.num_epochs):

            print("Iteration",epoch+1)

            samples = X.shape[0]

            for batch in range(0, samples, self.batch_size):
                print("\tBatch",1+(batch//self.batch_size), end=': ')

                D = {}
                for i in range(1,self.n_layers):
                    D[i]=np.zeros(self.w[i].shape)
                
                indices = range(batch, batch+self.batch_size)
                ypred = []

                for i in indices:
                    a = {}
                    z = {}
                    a[1] = X[i:i+1].T #column-vector
                    # forward
                    for layer in range(1,self.n_layers):
                        a[layer] = np.insert(a[layer], 0, np.ones(1), axis=0)   # insert bias node
                        z[layer+1] = np.dot(self.w[layer], a[layer])    # Z_l+1 = W_l.A_l
                        if 1+layer==self.n_layers:                      # A_l+1 = act_fn( Z_l+1 )
                            a[layer+1] = self.softmax(z[layer+1])
                        else:
                            a[layer+1] = self.act_fn(z[layer+1]) 
                    
                    ypred.append(np.argmax(a[self.n_layers]))

                    delta = {}
                    delta[self.n_layers] = a[self.n_layers] - y_[i:i+1].T

                    # backward
                    for layer in range(self.n_layers-1,0,-1):
                        if layer>1:
                            delta[layer] = (np.dot( self.w[layer].T , delta[layer+1] )[1:,:])*self.act_grad(z[layer])  #slicing to remove extra calculated error for bias
                        D[layer] += np.dot(delta[layer+1], a[layer].T)

                for i in range(1,self.n_layers):
                    print("Layer",i,self.w[i])
                    print()
                
                acc = np.mean(ypred==y[indices])
                print("Train Acc (for this batch) =",acc)

                for i in range(1,self.n_layers):
                    D[i]*=(self.learning_rate/samples)
                    # print("\t",i,np.sum(D[i]==0),D[i].shape[0]*D[i].shape[1])
                    self.w[i] -= D[i]
            

            if save_error:
                self.train_CE.append(self.CE_batch(y_,self.predict_proba(X)))
                if (Xtest is not None) and (ytest is not None):
                    self.test_CE.append(self.CE_batch(ytest_,self.predict_proba(Xtest)))
                    print("train CE:",self.train_CE[-1],"test CE:",self.test_CE[-1])

            

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """
        y = []

        for i in range(X.shape[0]):
            a={}
            z={}
            a[1] = X[i:i+1].T #column-vector

            for layer in range(1,self.n_layers):
                a[layer] = np.insert(a[layer], 0, np.ones(1), axis=0)   # insert bias node
                z[layer+1] = np.dot(self.w[layer], a[layer])    # Z_l+1 = W_l.A_l
                if 1+layer==self.n_layers:                      # A_l+1 = act_fn( Z_l+1 )
                    a[layer+1] = self.softmax(z[layer+1])
                else:
                    a[layer+1] = self.act_fn(z[layer+1])
            
            y.append(a[self.n_layers].reshape(-1))

        # return the numpy array y which contains the predicted values
        return np.array(y)

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        # return the numpy array y which contains the predicted values
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """
        ypred = self.predict(X)
        # return the numpy array y which contains the predicted values
        return np.mean(ypred==y)



