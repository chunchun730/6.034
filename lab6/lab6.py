# MIT 6.034 Lab 6: Neural Nets
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), Jake Barnwell (jb16), and 6.034 staff

from nn_problems import *
from math import e
INF = float('inf')

#### NEURAL NETS ###############################################################

# Wiring a neural net

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x >= threshold:
        return 1
    else:
        return 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1./(1.+e**(-(steepness*(x- midpoint))))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    if x < 0:
        return 0
    return x

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5*(actual_output - desired_output)**2

# Forward propagation

def node_value(node, input_values, neuron_outputs):  # STAFF PROVIDED
    """Given a node, a dictionary mapping input names to their values, and a
    dictionary mapping neuron names to their outputs, returns the output value
    of the node."""
    if isinstance(node, basestring):
        return input_values[node] if node in input_values else neuron_outputs[node]
    return node  # constant input, such as -1

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    neuron_out = {}
    final = 0
    for node in net.topological_sort():
        out = 0
        for i in net.get_incoming_neighbors(node):
            val = node_value(i, input_values, neuron_out)
            out += net.get_wires(i, node)[0].get_weight() * val
        neuron_out[node] = threshold_fn(out)
    return neuron_out[net.get_output_neuron()], neuron_out

# Backward propagation warm-up
def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    maximum = -INF
    best = []
    best, maximum = step(func, inputs, step_size, -INF, [], 0)
    return maximum, best

def step(func, inputs, step_size, maximum, best, index):
    if index == len(inputs):
        return is_better_fn(inputs, maximum, best, func)

    sizes = [0, step_size, -step_size]
    i = index
    for size in sizes:
        inputs[i] = inputs[i] + size
        best, maximum = step(func, inputs, step_size, maximum, best, i+1)
        inputs[i] = inputs[i] - size
    return best, maximum

def is_better_fn(inputs, maximum, best, func):
    if func(*inputs) > maximum:
        return inputs[:], func(*inputs)
    return best, maximum

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    end = wire.endNode
    start = wire.startNode
    res = set()
    res.add(end)
    res.add(start)
    res.add(wire)
    if net.is_output_neuron(end):
        return res
    else:
        queue = [end]
        while not net.is_output_neuron(queue[0]):
            out = queue.pop(0)
            for node in net.get_outgoing_neighbors(out):
                res.add(node)
                res.add(net.get_wires(out, node)[0])
                queue.append(node)
        res.add(queue[0])
    return res

# Backward propagation
def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    out_neuron = net.get_output_neuron()
    diff = desired_output - neuron_outputs[out_neuron]
    delta_Bs = {out_neuron: neuron_outputs[out_neuron] * (1-neuron_outputs[out_neuron]) * diff}
    graph = net.topological_sort()
    inputs = net.inputs

    for i in range(len(graph)-1, -1, -1):
        neuron = graph[i]
        for wire in net.get_wires(startNode = None, endNode = neuron):
            start = wire.startNode
            if start not in inputs:
                delta = 0
                for dep in get_back_prop_dependencies(net, wire):
                    if isinstance(dep, Wire):
                        start_ = dep.startNode
                        end_ = dep.endNode
                        if start_ == start and end_ in neuron_outputs:
                            if start in delta_Bs:
                                delta_Bs[start] += neuron_outputs[start_] * (1-neuron_outputs[start_]) * dep.get_weight() * delta_Bs[end_]
                            else:
                                delta_Bs[start] = neuron_outputs[start_] * (1-neuron_outputs[start_]) * dep.get_weight() * delta_Bs[end_]
    print delta_Bs
    return delta_Bs

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""

    delta_Bs = calculate_deltas(net, desired_output, neuron_outputs)
    graph = net.topological_sort()
    for neuron in graph:
        for incoming in net.get_incoming_neighbors(neuron):
            wire = net.get_wires(incoming, neuron)[0]
            if incoming in input_values:
                new_weight = wire.get_weight() + delta_Bs[neuron] * input_values[incoming] * r
            elif isinstance(incoming, (int, float)):
                new_weight = wire.get_weight() + delta_Bs[neuron] * incoming * r
            else:
                new_weight = wire.get_weight() + delta_Bs[neuron] * neuron_outputs[incoming] * r
            wire.set_weight(new_weight)
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    accuracy_ = -INF
    count = 0
    final_z, neuron_outputs = forward_prop(net, input_values, sigmoid)
    accuracy_ = accuracy(desired_output, final_z)
    while accuracy_ < minimum_accuracy:
        net = update_weights(net, input_values, desired_output, neuron_outputs, r)
        final_z, neuron_outputs = forward_prop(net, input_values, sigmoid)
        accuracy_ = accuracy(desired_output, final_z)
        count += 1
    return net, count


# Training a neural net

ANSWER_1 = 11
ANSWER_2 = 11
ANSWER_3 = 2
ANSWER_4 = 60
ANSWER_5 = 10

ANSWER_6 = 1
ANSWER_7 = 'checkerboard'
ANSWER_8 = ['small', 'medium', 'large']
ANSWER_9 = 'b'

ANSWER_10 = 'd'
ANSWER_11 = 'AC'
ANSWER_12 = 'AE'


#### SURVEY ####################################################################

NAME = 'Chunchun Wu'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = '8'
WHAT_I_FOUND_INTERESTING = 'None'
WHAT_I_FOUND_BORING = 'None'
SUGGESTIONS = 'None'
