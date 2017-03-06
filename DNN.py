import cPickle
import numpy as np


def unpickle(fileName):
    '''
    Description: retrieve data from CIFAR-10 Pickles
    Params: fileName = filename to unpickle
    Outputs: Unpickled Dict
    '''
    with open(fileName, 'rb') as f:
        dict = cPickle.load(f)
    return dict


def merge_batches(num_to_load=1):
    '''
    Description: Merge batches of CIFAR-10 data pickles
    Params: num_to_load = number of batches of CIFAR-10 to load and merge
    Outputs: merged features and labels from specified no. of batches of CIFAR-10
    '''
    for i in range(1):
        fileName = "cifar-10-batches-py/data_batch_" + str(i + 1)
        data = unpickle(fileName)
        if i == 0:
            features = data["data"]
            labels = np.array(data["labels"])
        else:
            features = np.append(features, data["data"], axis=0)
            labels = np.append(labels, data["labels"], axis=0)
    return features, labels


def one_hot_encode(data):
    '''
    Description: Encode Target Label IDs to one hot vector of size L where L is the
    number of unique labels
    Params: data = list of label IDs
    Outputs: List of One Hot Vectors
    '''
    one_hot = np.zeros((data.shape[0], 10))
    one_hot[np.arange(data.shape[0]), data] = 1
    return one_hot


def normalize(data):
    '''
    Description: Normalize Pixel values
    Params: list of Image Pixel Features
    Outputs: Normalized Image Pixel Features
    '''
    return data / 255.0


def preprocess(num_to_load=1):
    '''
    Description: helper function to load and preprocess CIFAR-10 training data batches
    Params: num_to_load = number of batches of CIFAR-10 to load and merge
    Outputs: Pre-processed CIFAR-10 image features and labels
    '''
    X, y = merge_batches(num_to_load=1)
    X = normalize(X)
    X = X.reshape(-1, 3072, 1)
    y = one_hot_encode(y)
    y = y.reshape(-1, 10, 1)
    return X, y


def dataset_split(X, y, ratio=0.8):
    '''
    Description: helper function to split training data into training and validation
    Params: X=image features
            y=labels
            ratio = ratio of training data from total data
    Outputs: training data (features and labels) and validation data
    '''
    split = int(ratio * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    training_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = X[training_idx, :], X[val_idx, :]
    y_train, y_val = y[training_idx, :], y[val_idx, :]
    print "Records in Training Dataset", X_train.shape[0]
    print "Records in Validation Dataset", X_val.shape[0]
    return X_train, y_train, X_val, y_val


def sigmoid(out):
    '''
    Description: Sigmoid Activation
    Params: out = a list/matrix to perform the activation on
    Outputs: Sigmoid activated list/matrix
    '''
    return 1.0 / (1.0 + np.exp(-out))


def delta_sigmoid(out):
    '''
    Description: Derivative of Sigmoid Activation
    Params: out = a list/matrix to perform the activation on
    Outputs: Delta(Sigmoid) activated list/matrix
    '''
    return sigmoid(out) * (1 - sigmoid(out))
	
def SigmoidCrossEntropyLoss(a, y):
        """
		Description: Calculate Sigmoid cross entropy loss
		Params: a = activation
				y = target one hot vector
		Outputs: a loss value
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


class DNN(object):
    '''
            Description: Class to define the Deep Neural Network
    '''

    def __init__(self, sizes):
        '''
        Description: initialize the biases and weights using a Gaussian
        distribution with mean 0, and variance 1.
        Biases are not set for 1st layer that is the input layer.
        Params: sizes = a list of size L; where L is the number of layers
                        in the deep neural network and each element of list contains
                        the number of neuron in that layer.
                        first and last elements of the list corresponds to the input
                        layer and output layer respectively
                        intermediate layers are hidden layers.
        '''
        self.num_layers = len(sizes)
        # setting appropriate dimensions for weights and biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        '''
        Description: Forward Passes an image feature matrix through the Deep Neural
                                 Network Architecture.
        Params: x = Image Features
        Outputs: 2 lists which stores outputs and activations at every layer,
                 1st list is non-activated and 2nd list is activated
                 The last element of the 2nd list corresponds to the scores against
                 10 labels in the dataset.
        '''
        activation = x
        activations = [x]  # list to store activations for every layer
        outs = []  # list to store out vectors for every layer
        for b, w in zip(self.biases, self.weights):
            out = np.dot(w, activation) + b
            outs.append(out)
            activation = sigmoid(out)
            activations.append(activation)
        return outs, activations

    def get_batch(self, X, y, batch_size):
        '''
        Description: A data iterator for batching of image features and labels
        Params: X, y = lists of Features and corresponding labels, these lists
                                   have to be batched.
                batch_size =  size of the batch
        Outputs: a batch of image features and labels of size = batch_size
        '''
        for batch_idx in range(0, X.shape[0], batch_size):
            batch = zip(X[batch_idx:batch_idx + batch_size],
                        y[batch_idx:batch_idx + batch_size])
            yield batch

    def train(self, X, y, batch_size=100, learning_rate=0.2, epochs=1000):
        '''
        Description: Batch-wise trains image features against corresponding labels.
                     The weights and biases of the neural network are updated through
                     backpropagation on batches using SGD
                     del_b and del_w are of same size as all the weights and biases
                     of all the layers. del_b and del_w contains the gradients which
                     are used to update weights and biases

        Params: X, y = lists of training features and corresponding labels
                batch_size =  size of the batch
                learning_rate = eta; controls the size of changes in weights & biases
                epochs = no. of times to iterate of the whole data
        '''
        n_batches = X.shape[0] / batch_size
        for j in xrange(epochs):
            batch_iter = self.get_batch(X, y, batch_size)
            for i in range(n_batches):
                batch = batch_iter.next()
                # same shape as self.biases
                del_b = [np.zeros(b.shape) for b in self.biases]
                # same shape as self.weights
                del_w = [np.zeros(w.shape) for w in self.weights]
                for batch_X, batch_y in batch:
                    # accumulate all the bias and weight gradients
                    loss, delta_del_b, delta_del_w = self.backpropagate(
                        batch_X, batch_y)
                    del_b = [db + ddb for db, ddb in zip(del_b, delta_del_b)]
                    del_w = [dw + ddw for dw, ddw in zip(del_w, delta_del_w)]
            # update weight and biases by multiplying ratio learning rate and batch_size
            # multiplied with the accumulated gradients(partial derivatives)
            # calculate change in weight(delta) and biases and update weight
            # with the changes
            self.weights = [w - (learning_rate / batch_size)
                            * delw for w, delw in zip(self.weights, del_w)]
            self.biases = [b - (learning_rate / batch_size)
                           * delb for b, delb in zip(self.biases, del_b)]
            print("\nEpoch %d complete\tLoss: %f\n"%(j, loss))

    def backpropagate(self, x, y):
		'''
		Description: Based on the derivative(delta) of cost function the gradients(rate of change 
					of cost function with respect to weights and biases) of weights
					and biases are calculated.
					del_b and del_w are of same size as all the weights and biases
					of all the layers. del_b and del_w are the gradients(partial derivatives)
					which are used to update the weights and biases.
					cost function here is SigmoidCrossEntropyLoss cost, hence cost_deriv is :
					delta C = activation(output_layer) - target
		Params: x, y = training feature and corresponding label (a training pair)
		Outputs: del_b: gradient of bias
				 del_w: gradient of weight
		'''
		del_b = [np.zeros(b.shape) for b in self.biases]
		del_w = [np.zeros(w.shape) for w in self.weights]

		# for calculating the current loss or cost forward pass through the neural net once
		# outs and activations are lists to store activations and out vector
		# for every layer
		outs, activations = self.feedforward(x)
		#Cost function:
		loss = SigmoidCrossEntropyLoss(activations[-1],y)
		# calculate derivative of cost Sigmoid Cross entropy which is to be minimized
		delta_cost = activations[-1] - y
		# backward pass to reduce cost
		# gradients at output layers
		delta = delta_cost
		del_b[-1] = delta
		del_w[-1] = np.dot(delta, activations[-2].T)

		# updating gradients of each layer using reverse or negative indexing, by propagating
		# gradients of previous layers to current layer so that gradients of weights and biases
		# at each layer can be calculated
		for l in xrange(2, self.num_layers):
			out = outs[-l]
			delta_activation = delta_sigmoid(out)
			delta = np.dot(self.weights[-l + 1].T, delta) * delta_activation
			del_b[-l] = delta
			del_w[-l] = np.dot(delta, activations[-l - 1].T)
		return (loss, del_b, del_w)

    def eval(self, X, y):
        '''
        Description: Based on trained(updated) weights and biases, predict a label and compare
                     it with original label and calculate accuracy
        Params: X, y = a data example from validation dataset (image features, labels)
        Outputs: accuracy of prediction
        '''
        count = 0
        for x, _y in zip(X, y):
            outs, activations = self.feedforward(x)
            # postion of maximum value is the predicted label
            if np.argmax(activations[-1]) == np.argmax(_y):
                count += 1
        print("Accuracy: %f" % ((float(count) / X.shape[0]) * 100))

    def predict(self, X):
        '''
        Description: Based on trained(updated) weights and biases, predict a label for an
                                 image which does not have a label.
        Params: X = list of features of unknown images
        Outputs: list containing the predicted label for the corresponding unknown image
        '''
        labels = unpickle("cifar-10-batches-py/batches.meta")["label_names"]
        preds = np.array([])
        for x in X:
            outs, activations = self.feedforward(x)
            preds = np.append(preds, np.argmax(activations[-1]))
        preds = np.array([labels[int(p)] for p in preds])
        return preds


def main():
    X, y = preprocess(num_to_load=1)
    X_train, y_train, X_val, y_val = dataset_split(X, y)
    # 32*32*3=3072, height and width of an image in the dataset is 32 and 3 is for RGB channel
    #[3072,1000,100,10] implies a neural network with 1 input layer of size 3072, 3 hidden
    # layers of size M, N and a output layer of size 10, hence 4
    # layers(including input layer), more layers can be added to the list for increasing layers
    model = DNN([3072, 50, 30, 10])  # initialize the model
    model.train(X_train, y_train, epochs=15)  # train the model
    model.eval(X_val, y_val)  # check accuracy using validation set
    # preprocess test dataset
    test_X = unpickle("cifar-10-batches-py/test_batch")["data"] / 255.0
    test_X = test_X.reshape(-1, 3072, 1)
    # make predictions of test dataset
    print model.predict(test_X)

main()
