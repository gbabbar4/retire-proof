import numpy as np


# Linear Class
class Linear():

  def __init__(self, inputs, output):

    self.inputs = inputs
    self.output = output

    # Initializing Random Values for Weight matrix with shape as (input, output)
    # np.random.normal(loc=0, scale=0.5,size=(self.inputs, self.output)) 
    # np.random.uniform(low = 0.1, high = 0.5, size = (self.inputs, self.output))
    self.weights = np.random.normal(loc=0, scale=0.5,size=(self.inputs, self.output)) 

    # Initializing Random Values for Bias matrix with shape as (ouptut , )
    self.bias = np.random.normal(loc = 0.1, scale = 0.2, size = (self.output, ))

    # To Store values from each neurons in a specific layer
    self.z = []

  def calc(self, input_list):

    self.z = np.matmul(input_list, self.weights) + self.bias
    self.inp_list = np.array(input_list)

    return self.z

  def backward(self, prev_grad):

    # Generic Gradient Calculation for the Weights
    # Z = (X * W^T ) + b
    self.dw = np.matmul(self.inp_list.T, prev_grad)

    # Generic Gradient Calculation for the Bias
    # ∂L/∂b = ∂L/∂z * ∂z/∂b --> general format
    # where ∂z/∂b = 1 , always
    self.db = prev_grad.sum(axis=0)

    # Generic Gradient Calculation for the Input values
    self.dX = np.matmul(prev_grad, self.weights.T)

    return self.dw, self.db , self.dX


  def update_parameters_linearlayer(self, eta):

    # Calculating the Updated values for
    # w :
    self.weights = self.weights - (eta * self.dw)
    # Re-Assigning the weights matrix with the updated weight matrix values

    # b :
    self.bias = self.bias - (eta * self.db)
    # Re-Assigning the Bias matrix with the updated Bias matrix values

    return self.weights, self.bias


############################################################################################

# ReLU Class
# @title
class ReLU():

  def calc(self, x):

    '''
    returns a final list which includes the ReLU values by intaking the Z Values
    '''

    self.final = np.ones(shape=(x.shape[0], x.shape[1]))
    row_len , col_len = x.shape[0], x.shape[1]

    for i in range(row_len):
        for j in range(col_len):
            if x[i][j] <= 0:
                self.final[i][j] = 0
            else:
                self.final[i][j] = x[i][j]

    return self.final

  def backward(self, prev_grad):

    '''
    prev_grad is a list containing gradients flowing into the ReLU from the next layer
    Note: ReLU value can be in the range of [0,Z]
    '''
    # length of self.final denotes the number of neurons
    # num_neurons = len(self.final)
    dL_dz = np.ones(shape=(self.final.shape[0], self.final.shape[1]))
    row_len , col_len = dL_dz.shape[0], dL_dz.shape[1]


    # # Calculating the Grad value for ∂L/∂z_21 & ∂L/∂z_22
    for i in range(row_len):
        for j in range(col_len):
            if self.final[i][j] <= 0:
                dL_dz[i][j] = 0
            else:
                dL_dz[i][j] = 1

    dr = dL_dz * prev_grad

    return dr


############################################################################################

# Loss Function
def Loss(y, y_hat):
  '''
  y : Target Value
  y_hat : Predicted Value
  '''
  return (sum((y_hat - y) ** 2) / len(y))[0]

# Sigmoid Function
class sigmoid():

  def calc(self, z):

    result = (1 / (1 + (np.e ** (- z))))

    return result
  
############################################################################################

# NeuralNetwork Class
class NeuralNetwork():

  def __init__(self, input_features, hidden_layer, num_hidden_layers):

      self.input_features = input_features
      self.hidden_layer = hidden_layer # Here `hidden_layer` denotes the number of number of neurons
      self.num_hidden_layers = num_hidden_layers # This denotes the number of Hidden Layers in the Network
      self.network_layers = {} # Ex format - {1 : [layer1, relu1], 2 : [layer2, relu2],....., n : [layern, relun]}

      layer1 = Linear(self.input_features, self.hidden_layer) # Layer 1
      relu1 = ReLU() # layer1 activation
      self.network_layers[0] = [layer1, relu1]

      for l in range(self.num_hidden_layers - 1): # Creates Hidden Layers

        layer_n = Linear(self.hidden_layer, self.hidden_layer) # Creating the Neurons in each Layer
        relu_n = ReLU() # Creating the ReLU Objects in each Layer

        self.network_layers[l + 1] = [layer_n, relu_n]

      layer_final_linear = Linear(self.hidden_layer, 1) # Layer N (Final Layer)
      sig = sigmoid() # Applying the sigmoid function for the last layer
      self.network_layers[self.num_hidden_layers] = [layer_final_linear, sig]


  def forward(self, input_list):

    # Layer info is stored in `self.network_layer` dict in the below format
    # {1 : [layer1, relu1], 2 : [layer2, relu2],....., n : [layern, sigmoidn]}

    l = len(self.network_layers)
    prev_relu_input_list = []


    layer = self.network_layers[0] #[0] = Linear Object ; [1] = ReLU Object
    # print(layer)
    # layer looks like this [<__main__.Linear object at 0x7e57dcafc9e0>, <__main__.ReLU object at 0x7e57b560b850>]

    layer_1 = layer[0].calc(input_list=input_list)
    # print(layer_1, layer_1.shape) # [0.15415477 0.12795178 0.1468451  0.12684121] (4,)

    relu_1 = layer[1].calc(layer_1)
    prev_relu_input_list.append(relu_1)

    for i in range(1, len(self.network_layers) - 1) :

      # self.network_layers[0,...,N] denotes the key value which is 1,2,3,...,N
      # which basically goes till the number of layers in the whole network

      layer = self.network_layers[i]
      # the above line returns a list - [layer_obj_{i} , ReLU_obj_{i}]
      # #[0] = Linear Object ; [1] = ReLU Object

      layer_n = layer[0].calc(input_list=prev_relu_input_list[-1]) # Linear Call
      relu_n = layer[1].calc(x=layer_n) # ReLU Call

      # Adding the relu output to the input list to be used by the next layer
      prev_relu_input_list.append(relu_n)


    # Making Forward pass for the last layer
    self.final_layer = self.network_layers[l - 1][0].calc(input_list=prev_relu_input_list[-1])
    # Sending the final layer output to sigmoid fn
    self.final_output = self.network_layers[l - 1][1].calc(self.final_layer)

    return self.final_output


  def backwards(self, y):

    # Calculating the Loss
    self.loss = Loss(self.final_output, y)

    # We will first Calculate ∂L/∂y_hat (change of Loss value with respect to the sigmoid value)
    # as We are applying the Sigmoid Activation function at the last layer
    y_hat = self.final_output
    l_y_hat = (2 * (y_hat - y)) / (y.shape[0]) # ∂L/∂y_hat - working correctly

    prev_grad_relu_bp = []
    l = len(self.network_layers)

    # Calculating ∂L/∂z_out -> ∂L/∂y_hat * ∂y_hat/∂z_out
    # where ∂y_hat/∂z_out = e^(z_out) / (1 + e^(z_out))^2, so in order to calculate ∂L/∂z_out
    # we require the raw output from the last layer
    
    sig_final = (np.e ** self.final_layer) / ((1 + (np.e ** self.final_layer)) ** 2) 
    dL_dz_out = l_y_hat * sig_final

    prev_grad_n = np.array(dL_dz_out)
    l3_w, l3_b, l3_X = self.network_layers[l - 1][0].backward(prev_grad=prev_grad_n)

    prev_grad_n1 = np.array(l3_X)
    # Calculating ∂L/∂r_{l-2} , ∂L/∂r_21 (in case 2 Hidden Layer)
    dL_dzn = self.network_layers[l - 2][1].backward(prev_grad=prev_grad_n1)

    prev_grad_relu_bp.append(dL_dzn)


    for i in range(l - 2, 0 , -1) :

      # Calling the Linear BP
      ln_w , ln_b, ln_X = self.network_layers[i][0].backward(prev_grad=prev_grad_relu_bp[-1])

      # Converting the ln_X as a numpy arr
      ln_X_np_arr = np.array(ln_X)

      # Callin the ReLU BP
      dL_dz_n = self.network_layers[i - 1][1].backward(prev_grad=ln_X_np_arr)

      # Appending the ReLU BP Output to the prev_grad list to be used by the next derivative chain
      prev_grad_relu_bp.append(dL_dz_n)


    # LAYER 1
    prev_grad1 = np.array(prev_grad_relu_bp[-1])
    l1_w, l1_b, l1_X = self.network_layers[0][0].backward(prev_grad=prev_grad1)


  def update_parameters(self, eta):
    '''
    Inputs :
      eta : Learning Rate

    Working :
      Calculates the Updated values of the Parameters using the Gradient Descent Algorithm
      x_new = x_old - eta(∂L/∂x_old)

    Outputs :
      List of Updated values for each and every parameter.
    '''
    
    l = len(self.network_layers)
    
    for i in range(l):

      self.network_layers[i][0].update_parameters_linearlayer(eta)
    


  def train_step(self, X, y, eta):

    # Forward Pass
    self.forward(input_list=X)

    # Predicted Values
    y_hat = self.final_output

    # Back-Propogation (Backward Pass)
    self.backwards(y=y)

    # Loss Calculation
    loss = self.loss

    # Parameter - Update (Gradient Descent Algorithm)
    self.update_parameters(eta=eta)

    return y_hat , loss


