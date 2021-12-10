import numpy as np
from math import sqrt
from math import exp

from numpy.lib.type_check import imag

def unpickle(file):
    """
    Method provided by CIFAR-10 data to unpack their images

    Parameters:
    file (str): Filepath

    Returns:
    dict: Returns a dictionary with both the images and labels associated with such
    """
    import pickle
    with open(file, 'rb') as input:
        dict = pickle.load(input, encoding='bytes')
    return dict

def config():
    """
    Layer details for the model. For every element of the array, if the first element is
    True, it is a convolutional (conv) layer. If it is false, then it a fully connected layer (FCL).
    If conv, the number in the second element describes the receptive field size of that layer.
    The third element then describes the number of filters to produce.
    If FCL, then the second number is meaningless and the third number lists the number of nuerons
    in the next hidden layer. This model must have at least 1 conv layer

    Return:
    list[list]: Description of layers
    """
    
    layers = [[True, 5, 5],
                [False, 0, 50]]

    return layers

def preprocess(data):
    """
    Sets up the image data in a manageable format and centers the data by subtracting the mean from
    each sample point.

    Parameters:
    data (list[list]): List containing image data

    Returns:
    list[np.array[32 x 32 x 3]]: List of images in a 32 x 32 x 3 format, centered around the mean. 
    """

    image_data = np.array(data[b'data'])
    image_data = (image_data - np.mean(image_data)) / np.std(image_data)
    images = []
    for sample in range(0, 10000):
        image = np.empty([32, 32, 3])
        for color in range(0, 3):
            for row in range(0, 32):
                start = color * 1024 + row * 32
                end = start + 32
                image[row, :, color] = image_data[sample][start:end]
        images.append(image)
    
    return images

def rot_images(images):
    """
    Creates a list of images rotated by factors of 90 degrees.

    Parameters:
    images (list[np.array[32 x 32 x 3]]): List of image representations.

    Returns:
    list[np.array[32 x 32 x 3]]: List of 90 degree rotated image representations.
    list[int]: Labels for rotated images. Multiply by 90 to obtain degrees.
    list[np.array[32 x 32 x 3]]: List of 180 degree rotated image representations.
    list[int]: Labels for rotated images. Multiply by 180 to obtain degrees.
    """

    n = len(images)
    rot90_images = []
    rot90_labels = []
    current_image = np.array((32, 32, 3))

    for i in range(0, n):
        current_image = images[i]
        rot90_images.append(current_image)
        rot90_labels.append(int(0))

        for j in range(1, 4):
            current_image = np.rot90(current_image)
            rot90_images.append(current_image)
            rot90_labels.append(int(j))

    rot180_images = rot90_images[::2]
    rot180_labels = rot90_labels[::2]
    
    for i in range(1, int(n * 2), 2):
        rot180_labels[i] = int(rot180_labels[i] / 2)
   
    return rot90_images, rot90_labels, rot180_images, rot180_labels

def init_network(layers):
    """
    Initializes the appropriate weight vectors based on the parameters of the model. For
    no, the parameters are decided mannually in this method.

    Returns:
    numpy array [5 x 5 x 5]: Arrays of weights for each filter.
    numpy array [5]: Array of bias values.
    """
    N = len(layers)
    H = 32
    W = []
    B = []
    D = 3
    
    for i in range(0, N):
        if layers[i][0] == True:                            # If CONV
            w_conv = []
            b_conv = []
            F = layers[i][1]
            H = H - F + 1
            K = layers[i][2]
            for j in range(0, K):
                w_conv.append(np.random.randn(F, F, D) * sqrt(2 / (F * F * D)))
                b_conv.append(np.random.randn() * sqrt(2 / (F * F * D)))
            D = K
            W.append(w_conv)
            B.append(np.array(b_conv))
        elif layers[i - 1][0] == True:                      # If first FCL
            W.append(np.random.randn(layers[i][2], H * H * D) * sqrt(2 / H * H * D))
            B.append(np.random.randn(layers[i][2]) * sqrt(2 / H * H * D))
        else:                                               # Intermediate FCL
            W.append(np.random.randn(layers[i][2], layers[i - 1][2]) * sqrt(2 / layers[i - 1][2]))
            B.append(np.random.randn(layers[i][2]) * sqrt(2 / layers[i - 1][2]))

    w_sup = np.random.randn(10, layers[N - 1][2]) * sqrt(2 / layers[N - 1][2])
    w_rot90 = np.random.randn(4, layers[N - 1][2]) * sqrt(2 / layers[N - 1][2])
    w_rot180 = np.random.randn(2, layers[N - 1][2]) * sqrt(2 / layers[N - 1][2])
    b_sup = np.random.randn(10) * sqrt(2 / layers[N - 1][2])
    b_rot90 = np.random.randn(4) * sqrt(2 / layers[N - 1][2])
    b_rot180 = np.random.randn(2) * sqrt(2 / layers[N - 1][2])  
    W_end = ([w_sup, w_rot90, w_rot180])
    B_end = ([b_sup, b_rot90, b_rot180])
    
    return W, B, W_end, B_end

def conv(input, W, B):
    """
    Performs convolution on a matrix to reduce the dimensionality of the output.

    Parameters:
    input (numpy.array[32 * 32 * 5]): Matrix representation of an image.
    W: (numpy.array[F * F * K]): Filter matrices.
    b: (numpy.array[K x 1]): Bias values for each filters.
    """

    D = input.shape[2]
    H_1 = input.shape[0]
    K = len(W)
    F = W[0].shape[0]
    H_2 = H_1 - F + 1
    A = np.empty([H_2, H_2, F])

    for filter in range(0, K):
        for column in range(0, H_2):
            for row in range(0, H_2):
                sum = 0
                for depth in range(0, D):
                    sum = sum + input[row:row + F, column:column + F, depth].flatten().dot(W[filter][:,:,depth].flatten()) + B[filter]
                
                if sum > 0:
                    A[row, column, filter] = sum + B[filter]
                else:
                    A[row, column, filter] = 0
    
    return A

def FCL(input, W, b):
    """
    Creates the fully connected layers for a multi-dimensional image representation.

    Parameters:
    input (np.array): Output of a convolutional layer.
    W (np.array): Weight matrix for the transformation.
    b (np.array): Bias array for the transformation.

    Returns:
    np.array([n x 1]): One dimensional array of output layer.
    """

    A = W.dot(input.flatten()) + b
    n = len(A)
    for i in range(0, n):
        if A[i] < 0:
            A[i] = 0

    return A

def softmax(O):
    """
    Implements the softmax algorithm on the last hidden layer.

    Parameters:
    O (np.array): One-dimensional utput of last hidden layer.

    Returns:
    np.array: Normalized output layer.
    """

    sum = 0
    n = len(O)
    logC = O.max()
    V = np.empty(n)
    for i in range(0, n):
        sum = sum + exp(O[i] - logC)
    
    for i in range(0, n):
        V[i] = (exp(O[i] - logC)) / sum
    
    return V

def apply_net(input, W, B):
    """
    Calculates the output for a single sample image.

    Parameters:
    input (np.array [32 x 32 x 5]): Image representation.
    W (np.array): Weight matrices for each layer.
    B (np.array): Bias arrays for each layer.

    Returns:
    list[np.array]: Calculated values for each layer.
    list: Output of model.
    """
    
    N = len(W)
    A = []
    A.append(conv(input, W[0], B[0]))
    for i in range(1, N):
        if type(W[i]) == list:
            A.append(conv(A[i - 1], W[i], B[i]))
        else:
            A.append(FCL(A[i - 1], W[i], B[i]))
    
    y = softmax(A[N - 1])
    
    return A, y

def loss(y, label):
    """
    Calculates the loss for a single example.

    Paramters:
    y (list): Model label predictions.
    label (int): Actual label.
    """
    n = len(y)
    target = [0] * n
    target[label - 1] = 1
    L = 0
    for i in range(0, n):
        L = L + 0.5 * (target[i] - y[i]) ** 2
    
    return L

def total_loss(images, W, W_mode, B, B_mode, labels):
    """
    Calculates the total loss for the training set given the current model.

    Parameters:
    images (list[np.array [32 x 32 x 5]]): List of image representations
    W (np.array): Weight matrices for each layer.
    B (np.array): Bias arrays for each layer.
    labels (list): List of labels for each image.
    """
    
    temp_W = W[::]
    temp_B = B[::]
    temp_W.append(W_mode)
    temp_B.append(B_mode)
    L = 0
    n = len(images)
    for i in range(0, n):
        A, y = apply_net(images[i], temp_W, temp_B)
        L = L + loss(y, labels[i])
    
    L = L / n
    return L

def init_D(A):
    """
    Creates a list of empty matrices for to hold the error values for
    backpropagation.

    Paramters:
    A [list[np.array]]: Neuron values for the current forward pass

    Returns:
    list[np.array]: Returns a list of numpy matrices to store the error values.
    """

    D = []
    n = len(A)
    for i in range(0, n):
        D.append(np.empty(A[i].shape))

    return D

def d_soft(y, label):
    """
    Calculates the weighted error values for the softmax layer.

    Parameters:
    y (np.array[10 x 1]): Output of the model for the given sample.
    lab (int): Actual label for the sample.

    Returns:
    np.array[10 x 1]: Weighted error vector for the softmax layer.
    """
    
    n = len(y)
    d = np.empty(n)
    t = [0] * n
    t[label - 1] = 1
    for i in range(0, n):
        d[i] = (t[i] - y[i])
    
    return d

def d_FCL(a, d_j, w):
    """
    Calculates the weighted error vector for a FCL hidden layer.

    Parameters:
    a (np.array): Neuron values for the layer.
    d_j (np.array): Weighted error values for the next forward layer.
    w (np.array): Weights for the transform between this and next forward
                    layer.

    Returns:
    np.array: Weighted error values for layer.
    """
    n = w.shape[1]

    if len(a.shape) == 3:
        temp = a.flatten()
    else:
        temp = a

    d_i = np.transpose(w).dot(d_j)
    for i in range(0, n):
        if temp[i] == 0:
            d_i[i] = 0

    return d_i

def d_CONV(a, d_j, w):
    """
    Calculates the weighted error matrix for a CONV hidden layer.

    Paramters:
    a (np.array): Neuron values for the layer.
    d_j (np.array): Weighted error values for the next forward layer.
    w (list[np.array]): Weights for the transform between this and next forward
                    layer.
    """

    temp = a.flatten()
    H_i = a.shape[0]
    n_i = H_i ** 2
    D = a.shape[2]
    F = w[0].shape[0]
    K = w[0].shape[2]
    H_j = H_i - F + 1
    n_j = H_j ** 2
    d_i = np.zeros([n_i, D])

    for filter in range(0, K):                              # Filter
        filter_start = n_j * filter
        for i in range(0, H_j):                             # Output volume row
            for j in range(0, H_j):                         # Output volume column
                out_index = filter_start + H_j * i + j
                o = d_j[out_index]
                for depth in range(0, D):                   # Input depth
                    depth_start = n_i * depth
                    for k in range(0, F):                   # Input row
                        row = (i + k) * H_i
                        for m in range(0, F):               # Input col
                            in_index = row + j + m          # Input row
                            if temp[filter * n_i + in_index] > 0:
                                d_i[in_index, filter] = d_i[in_index, filter] + o * w[filter][k, m, depth]

    return d_i

def backprop(A, y, label, W):
    """
    Creates the weighted error matrices from the values of the neurons and
    the current model.

    Parameters:
    A (list[np.array]): List of arrays describing the values of each neuron.
    y (np.array): Output layer of the model for the given sample.
    label (int): Actual label for the given sample.
    W (list[np.array]): Weight matrices for each layer.

    Returns:
    list[np.array]: Weighted error matrices for each layer.
    """
    
    D = init_D(A)
    n = len(A) - 1
    D[n] = d_soft(y, label)
    for i in range(n - 1, -1, -1):
        if type(W[i + 1]) != list:
                D[i] = d_FCL(A[i], D[i + 1], W[i + 1])
        else:
            if len(D[i + 1].shape) == 1:
                D[i] = d_CONV(A[i], D[i + 1], W[i + 1])
            else:
                D[i] = d_CONV(A[i], D[i + 1].flatten(), W[i + 1])
    
    return D

def init_grad(W):
    """
    Initializes a list of matrices with all zeros to be filled.

    Parameters:
    W (list[np.array]): Weight matrices for each layer.

    Returns:
    (list[np.array]): List of zeroed matrices.
    """
    
    dW = []
    dB = []

    for i in W:
        if type(i) == list:
            temp = []
            for j in i:
                temp.append(np.zeros(j.shape))
            dW.append(temp)
            dB.append(np.zeros(len(i)))
        else:
            dW.append(np.zeros(i.shape))
            dB.append(np.zeros(i.shape[0]))

    return dW, dB

def grad_CONV(a, d, dW, dB):
    """
    Computes the gradient of the weight matrix for a CONV operation

    Parameters:
    a [np.array]: Neuron values.
    d [np.array]: Weight error matrix.
    dW [list[np.array]]: List of error matrices for each filter.

    Returns:
    list[np.array]: Gradient values for each filter.
    """

    count = 0
    H_1 = a.shape[0]
    n_1 = H_1 * H_1
    D = a.shape[2]
    F = dW[0].shape[0]
    H_2 = H_1 - F + 1
    if len(d.shape) == 1:
        d = d.reshape((H_2 * H_2, F))
    K = d.shape[1]
    n_2 = d.shape[0]

    for filter in range(0, K):
        filter_start = filter * n_2
        for i in range(0, H_2):                 # Output volume row
            for j in range(0, H_2):             # Output volume column
                out_index = i * H_2 + j
                o = d[out_index, filter]
                for depth in range (0, D):
                    for k in range(0, F):
                        for m in range(0, F):
                            change = a[k, m, depth] * o
                            dW[filter][k, m, depth] = dW[filter][k, m, depth] + change
                            dB[filter] = dB[filter] + change
                            count = count + 1
    
    for i in range(0, K):
        dW[i] = dW[i] / n_2
        dB[i] = np.sum(dW[i]) / dW[i].size

    return dW, dB

def grad_descent(x, A, D, W):
    """
    Performs gradient descent on the entire model for the given sample.

    Parameters:
    x (np.array): Input image representation.
    A (list[np.array]): Neuron values for each layer.
    D (list[np.array]): Weighted error matrices for each layer.
    W (list[np.array]): Model parameters.

    Returns:
    list[np.array]: Gradient values for each weight matrix.
    """
    
    dW, dB = init_grad(W)
    n = len(dW)

    dW[0], dB[0] = grad_CONV(x, D[0], dW[0], dB[0])

    for i in range(1, n):
        if type(dW[i]) == list:
            dW[i], dB[i] = grad_CONV(A[i - 1], D[i], dW[i], dB[i])
            
        else:
            if len(A[i - 1].shape) == 3:
                temp = np.asmatrix(A[i - 1].flatten())
                dW[i] = np.transpose(np.asmatrix(D[i])).dot(temp)
            else:
                dW[i] = np.transpose(np.asmatrix(D[i])).dot(np.asmatrix(A[i - 1]))
            dB[i] = D[i]

    return dW, dB

def add_grad(dW, dB, change_W, change_B):
    """
    Changes the model by the amount specified by the gradient and learning rate.
    Default learning rate is set ot 0.1.

    Parameters:
    W (list[np.array]): Model parameters.
    dW (list[np.array]): Gradient values for each weight matrix.)
    """
    
    n = len(dW)

    for i in range(0, n):
        if type(dW[i] == list):
            m = len(dW[i])
            for j in range(0, m):
                dW[i][j] = dW[i][j] + change_W[i][j]
                dB[i][j] = dB[i][j] + change_B[i][j]
        else:
            dW[i] = dW[i] + change_W[i]
            dB[i] = dB[i] + change_B[i]

    return dW, dB

def update_w(W, B, dW, dB, num_samples, rate = 0.1):
    """
    Changes the model by the amount specified by the gradient and learning rate.
    Default learning rate is set ot 0.1.

    Parameters:
    W (list[np.array]): Model parameters.
    dW (list[np.array]): Gradient values for each weight matrix.)
    """

    n = len(W)

    for i in range(0, n):
        if type(W[i] == list):
            m = len(W[i])
            for j in range(0, m):
                W[i][j] = W[i][j] - ((dW[i][j] * rate) / num_samples)
                B[i][j] = B[i][j] - ((dB[i][j] * rate) / num_samples)
        else:
            W[i] = W[i] - ((dW[i] * rate) / num_samples)
            B[i] = B[i] - ((dB[i] * rate) / num_samples)

    return W, B

def train(X, W, W_mode, B, B_mode, labels, rate = 0.1):
    """
    Performs training given a set of images. Uses a stochastic gradient
    descent algorithm and returns the updated model.

    Parameters:
    X (list[numpy.array[32 x 32 x 5]]): Image data.
    W (list[numpy.array]): Model weight parameters.
    W_mode (numpy.array): Last weight layer for the specific task.
    B (list[numpy.array]): Model bias parameters.
    B_mode (numpy.array): Last bias layer for the specific task.
    labels (list[int]): Pair of lists containing the labels for both the supervised and rotational
                                tasks. The supervised labels must be first and the rotational second.
    rate (float): Learning rate.

    Returns:
    list[numpy.array]: Model weight parameters.
    numpy.array: Last weight layer for the specific task.
    list[numpy.array]: Model bias parameters.
    numpy.array: Last bias layer for the specific task.
    """
    
    W.append(W_mode)
    B.append(B_mode)
    n = len(X)
    m = len(W) - 1

    dW, dB = init_grad(W)
    for i in range(0, n):
        A, y = apply_net(X[i], W, B)
        D = backprop(A, y, labels[i], W)
        change_W, change_B = grad_descent(X[i], A, D, W)
        dW, dB = add_grad(dW, dB, change_W, change_B)

    W, B = update_w(W, B, dW, dB, n, rate)
    
    return W[:-1], B[:-1], W[m], B[m]

def train_model(X, W, W_mode, B, B_mode, labels, rate = 0.1, weight = 0.1, max_iter = 100):
    """
    Trains the model with both the rotational pretext task and then the supervised method.

    Parameters:
    X (list[list[numpy.array[32 x 32 x 5]]]): Pair of lists containing the image data. The first
                                                list must be the unaltered images and the second
                                                the rotated images.
    W (list[numpy.array]): Model weight parameters.
    W_mode (list[numpy.array]): Pair of arrays containing the last layer network weights for both the
                                supervised learning and the rotational task. The supervised weights
                                must be first in the list and the rotational second.
    B (list[numpy.array]): Model bias parameters.
    B_mode (list[numpy.array]): Pair of arrays containing the last layer network bias for both the
                                supervised learning and the rotational task. The supervised bias
                                must be first in the list and the rotational second.
    labels (list[list[int]]): Pair of lists containing the labels for both the supervised and rotational
                                tasks. The supervised labels must be first and the rotational second.
    rate (float): Learning rate.
    weight (float): Pretext task loss weight.
    """
    L_rot = []
    L_sup = []
    print(total_loss(X[0], W, W_mode[0], B, B_mode[0], labels[0]) + weight * total_loss(X[1], W, W_mode[1], B, B_mode[1], labels[1]))

    for epoch in range(0, max_iter):
        W, B, W_mode[1], B_mode[1] = train(X[1], W, W_mode[1], B, B_mode[1], labels[1], rate)       # Train rotational set
        L_rot.append(total_loss(X[1], W, W_mode[1], B, B_mode[1], labels[1]))
        L_sup.append(total_loss(X[0], W, W_mode[0], B, B_mode[0], labels[0]))
        print("Rotational Loss: " + str(L_rot[epoch]) + "\tTotal Loss: " + str(L_sup[epoch] + weight * L_rot[epoch])) 
    for epoch in range(0, max_iter):
        W, B, W_mode[0], B_mode[0] = train(X[0], W, W_mode[0], B, B_mode[0], labels[0])       # Train supervised set
        L_rot.append(total_loss(X[1], W, W_mode[1], B, B_mode[1], labels[1]))
        L_sup.append(total_loss(X[0], W, W_mode[0], B, B_mode[0], labels[0]))
        print("Supervised Loss: " + str(L_sup[epoch]) + "\tTotal Loss: " + str(L_sup[epoch] + weight * L_rot[epoch])) 

    return W, B, W_mode, B_mode, L_rot, L_sup

def test(X, W, B, labels):
    """
    For a given test set, displays the percentage of correct predictions to the standard
    output.

    Parameters:
    X (list[np,array]): List of image representations.
    W (list[np.array]): Model weight parameters.
    B (list[np.array]): Model bias parameters.
    labels (list[int]): Actual image labels.
    """

    n = len(X)
    correct = 0
    for i in range(0, n):
        A, y = apply_net(X[i], W, B)
        if (np.argmax(y) - 1) == labels[i]:
            correct = correct + 1
    
    print("The model predicts " + str(correct / n) + "% of image classes correctly")

def save(L_rot, L_sup, weight, filename):
    """
    Saves the data to a csv file as specified by the given filename.

    Parameters:
    L_rot (np.array): Loss data for the rotational pretext task.
    L_sup (np.array): Loss data for the supervised learning task.
    weight (float): Weight used in the overall loss function.
    filename (str): Filepath in which to save the csv file.
    """

    n = len(L_rot)
    with open(filename, 'w') as file:
        file.write("Rotational, Supervised, Total\n")
        for i in range(0, n):
            file.write(str(L_rot[i]))
            file.write(',')
            file.write(str(L_sup[i]))
            file.write(',')
            file.write(str(L_sup[i] + weight * L_rot[i]))
            file.write('\n')

"""
MAIN DRIVER CODE
"""

data = unpickle("../data/cifar-10-batches-py/data_batch_1")
images = preprocess(data)
labels = data[b'labels']
layers = config()
images_90, labels_90, images_180, labels_180 = rot_images(images)
W, B, W_mode, B_mode = init_network(layers)
W, B, W_mode, B_mode, L_rot, L_sup = train_model([images[0:100], images_180[400:4000]], W, [W_mode[0], W_mode[1]], B, [B_mode[0], B_mode[1]], [labels[0:100], labels_180[400:4000]], -0.0001, 0.1, 100)
W.append(W_mode[0])
B.append(B_mode[0])
save(L_rot, L_sup, 0.1, "../data/d90_10_3.csv")
test(images[2000:4000], W, B, labels[2000:4000])