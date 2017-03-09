import numpy as np

from cs231n.layers import *
from cs231n.layers import L2_loss
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    C,H,W=input_dim
    self.H1=(H-2)/2+1
    self.W1=(W-2)/2+1
    self.filt_num=num_filters
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1']=np.zeros(num_filters)
    self.params['W2']=weight_scale*np.random.randn(self.H1*self.W1*num_filters,hidden_dim)
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3']=np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    N=X.shape[0]
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_1,cache_1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    out_2,cache_2=affine_relu_forward(out_1,W2,b2)
    scores,cache_3=affine_forward(out_2,W3,b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,grad_x=softmax_loss(scores,y)
    loss=loss+0.5*self.reg*np.sum(W1*W1)+self.reg*np.sum(W2*W2)+self.reg*np.sum(W3*W3)
    grad_x,grads['W3'],grads['b3']=affine_backward(grad_x,cache_3)
    grad_x,grads['W2'],grads['b2']=affine_relu_backward(grad_x,cache_2)
    grad_x=grad_x.reshape(N,self.filt_num,self.H1,self.W1)
    _,grads['W1'],grads['b1']=conv_relu_pool_backward(grad_x,cache_1)
    grads['W1']+=self.reg*W1
    grads['W2']+=self.reg*W2
    grads['W3']+=self.reg*W3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
class FourLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    C,H,W=input_dim
    self.H1=(H-2)/2+1
    self.W1=(W-2)/2+1
    self.H2=(self.H1-2)/2+1
    self.W2=(self.W1-2)/2+1
    self.filt_num=num_filters
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1']=np.zeros(num_filters)
    self.params['W2']=weight_scale*np.random.randn(num_filters,num_filters,filter_size,filter_size)
    self.params['b2']=np.zeros(num_filters)
    self.params['W3']=weight_scale*np.random.randn(self.H2*self.W2*num_filters,hidden_dim)
    self.params['b3']=np.zeros(hidden_dim)
    self.params['W4']=weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b4']=np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    N=X.shape[0]
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_1,cache_1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    out_2,cache_2=conv_relu_pool_forward(out_1,W2,b2,conv_param,pool_param)
    out_3,cache_3=affine_relu_forward(out_2,W3,b3)
    scores,cache_4=affine_forward(out_3,W4,b4)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,grad_x=softmax_loss(scores,y)
    loss=loss+0.5*self.reg*np.sum(W1*W1)+self.reg*np.sum(W2*W2)+self.reg*np.sum(W3*W3)+self.reg*np.sum(W4*W4)
    grad_x,grads['W4'],grads['b4']=affine_backward(grad_x,cache_4)
    grad_x,grads['W3'],grads['b3']=affine_relu_backward(grad_x,cache_3)
    grad_x=grad_x.reshape(N,self.filt_num,self.H2,self.W2)
    grad_x,grads['W2'],grads['b2']=conv_relu_pool_backward(grad_x,cache_2)
    _,grads['W1'],grads['b1']=conv_relu_pool_backward(grad_x,cache_1)


    grads['W1']+=self.reg*W1
    grads['W2']+=self.reg*W2
    grads['W3']+=self.reg*W3
    grads['W4']+=self.reg*W4
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
class FiveLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    C,H,W=input_dim
    self.H1=(H-2)/2+1
    self.W1=(W-2)/2+1
    self.H2=(self.H1-2)/2+1
    self.W2=(self.W1-2)/2+1
    self.H3=(self.H2-2)/2+1
    self.W3=(self.W2-2)/2+1
    self.filt_num=num_filters
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1']=np.zeros(num_filters)
    self.params['W2']=weight_scale*np.random.randn(num_filters,num_filters,filter_size,filter_size)
    self.params['b2']=np.zeros(num_filters)
    self.params['W3']=weight_scale*np.random.randn(num_filters,num_filters,filter_size,filter_size)
    self.params['b3']=np.zeros(num_filters)
    self.params['W4']=weight_scale*np.random.randn(self.H3*self.W3*num_filters,hidden_dim)
    self.params['b4']=np.zeros(hidden_dim)
    self.params['W5']=weight_scale*np.random.randn(hidden_dim,num_classes)*0.001
    self.params['b5']=np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    N=X.shape[0]
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_1,cache_1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    out_2,cache_2=conv_relu_pool_forward(out_1,W2,b2,conv_param,pool_param)
    out_3,cache_3=conv_relu_pool_forward(out_2,W3,b3,conv_param,pool_param)
    out_4,cache_4=affine_relu_forward(out_3,W4,b4)
    scores,cache_5=affine_forward(out_4,W5,b5)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,grad_x=L2_loss(scores,y)
    loss=loss+0.5*self.reg*np.sum(W1*W1)+self.reg*np.sum(W2*W2)+self.reg*np.sum(W3*W3)+self.reg*np.sum(W4*W4)
    grad_x,grads['W5'],grads['b5']=affine_backward(grad_x,cache_5)
    grad_x,grads['W4'],grads['b4']=affine_relu_backward(grad_x,cache_4)
    grad_x=grad_x.reshape(N,self.filt_num,self.H3,self.W3)
    grad_x,grads['W3'],grads['b3']=conv_relu_pool_backward(grad_x,cache_3)
    grad_x,grads['W2'],grads['b2']=conv_relu_pool_backward(grad_x,cache_2)
    _,grads['W1'],grads['b1']=conv_relu_pool_backward(grad_x,cache_1)


    grads['W1']+=self.reg*W1
    grads['W2']+=self.reg*W2
    grads['W3']+=self.reg*W3
    grads['W4']+=self.reg*W4
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  