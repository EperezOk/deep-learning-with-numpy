import numpy as np

linear_dataset = np.array([
    { "data": [2, 3], "label": -1 },
    { "data": [4, 3], "label": -1 },
    { "data": [4, 4], "label": -1 },
    { "data": [3, 3.5], "label": -1 },
    { "data": [1.5, 3.5], "label": -1 },
    { "data": [1, 5], "label": 1 },
    { "data": [2, 5.5], "label": 1 },
    { "data": [3, 5], "label": 1 },
    { "data": [1.5, 4.5], "label": 1 },
    { "data": [3.5, 5.5], "label": 1 },
])

def get_linear_data():
    inputs = np.array([d["data"] for d in linear_dataset])
    outputs = np.array([d["label"] for d in linear_dataset])
    return inputs, outputs
