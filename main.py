'''
Optimization algo used to find local minimum,
Used in ML to opitimize weights and biases
'''

import numpy as np
import matplotlib.pyplot as plt

# Use the function x^2
def y_function(x):
    return x ** 2

def y_derivative(x):
    return 2 * x

x = np.arange(-100, 100, 0.1)
y = y_function(x)

current_pos = (80, y_function(80))

# Rate at which algo updates itself
learning_rate = 0.001

for _ in range(1000):
    # Minus because we are reversing direction
    new_x = current_pos[0] - learning_rate * y_derivative(current_pos[0])
    new_y = y_function(new_x)
    current_pos = (new_x, new_y)

plt.plot(x, y)
plt.scatter(current_pos[0], current_pos[1], color="red")
plt.pause(0.001)
plt.clf()