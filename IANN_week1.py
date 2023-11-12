
#%%
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

#%% Load Data
# 2.1 
# =========================
digits = load_digits()
print( digits.data.shape, digits.target.shape )

#%% One example
image_exp = digits.data[100]
target_exp = digits.target[100]
plt.imshow(image_exp.reshape(8,8), cmap="gray")
print(target_exp)

#%% Normalize Inputs
data = digits.data / 16.0
data.min(), data.max()

#%% One-Hot Targets
target = digits.target.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
target_onehot = enc.fit_transform(target).todense()

target.shape, target_onehot.shape 

#%% Generate Batches
class Batches:
    def __init__(self, data, target_onehot, batch_size):
        self.inputs = data
        self.targets = target_onehot
        self.batch_size = batch_size

    def generate_batches(self):
        n_images = len(self.inputs)
        batch_start = np.arange(0, n_images, self.batch_size)
        indices = np.arange(n_images, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.inputs), np.array(self.targets), batches

train_data = Batches(data, target_onehot, 50)
inputs, targets, batches = train_data.generate_batches()

#%% Test Batches
cnt = 0
for batch in batches:
    cnt += 1
    if cnt == 1 :
        x_batches = inputs[batch]
        y_batches = targets[batch]

cnt, x_batches.shape, y_batches.shape

#%% Sigmoid
# 2.2
# =========================
class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

sigmoid_func = Sigmoid()

# test
out = sigmoid_func(x_batches)

out.shape

#%% Softmax
# 2.3
# =========================
class Softmax:
    def __call__(self, x):
        exp_x = np.exp(x)
        return exp_x/exp_x.sum(axis=1,keepdims=True)

softmax_func = Softmax()

# test
out = softmax_func( np.random.rand(50, 10) )

out.shape

#%% MLP Layer
# 2.4
# =========================

class MLPLayer:
    def __init__(self, input_size, nb_units, active_func):
      
        self.input_size = input_size
        self.nb_units = nb_units
        self.active_func = active_func

        self.weights = np.random.normal(0.0, 0.2, size=(input_size, nb_units))
        self.bias = np.zeros((1, nb_units))

    def forward(self, x):

        out = np.dot(x, self.weights) + self.bias # Wx + b
        out = self.active_func(out) # sigmoid( Wx + b )

        return out

# test
layer_ = MLPLayer(input_size=64, nb_units=3, active_func=sigmoid_func)

out = layer_.forward(x_batches)

out.shape

#%% MLP Class
# 2.4
# =========================

class MLP:
    def __init__(self, nb_layers, layers_inputs, layers_units, layers_active_func):

        self.layers = []

        for i in range(nb_layers):
            layer_ = MLPLayer(layers_inputs[i], layers_units[i], layers_active_func[i])
            self.layers.append(layer_)

    def forward(self, x):

        for layer_ in self.layers:
            x = layer_.forward(x)

        return x

# test 
layers_inputs = [64, 128, 128]
layers_units = [128, 128, 10]
layers_active_func = [sigmoid_func, sigmoid_func, softmax_func]

network = MLP(3, layers_inputs, layers_units, layers_active_func)

probs_ = network.forward(x_batches)

probs_.shape

#%% MLP Class
# 2.5
# =========================

class CCELoss:
    def __call__(self, p_pred, y_true):
        loss = -np.sum(y_true * np.log(p_pred)) / len(y_true)
        return loss

# test
loss_func = CCELoss()

loss_ = loss_func(probs_, y_batches)

loss_

#%% Backprop

# TBC



# %%
