import numpy as np

inputs = [  # 3x4
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8],
]
weights = [  # 3x4
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]
biases = [2, 3, 0.5]

print('old:')
print(np.dot(weights, inputs[0]) + biases)
print()

print('new:')
output = np.dot(inputs, np.array(weights).T) + biases
print(output)
print()
