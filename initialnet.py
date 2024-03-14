import numpy as np

def initialnet(N, p, R, gamma):
    # Create random matrix where the left columns are excitatory neurons and 
    # right columns are inhibitory neurons with zeros on the diagonal and 
    # density p elsewhere.
    NN = round(p * N * (N - 1))
    fill = np.concatenate((np.ones(NN), np.zeros(N*(N - 1) - NN)))
    np.random.shuffle(fill)
    fill = np.reshape(fill, (N, N - 1))
    
    W1 = np.zeros((N, N))
    W1[:-1, 1:] = fill[:-1, :]
    
    W2 = np.zeros((N, N))
    W2[1:, :-1] = fill[1:, :]
    
    W = np.triu(W1, 1) + np.tril(W2, -1)
    
    # Create synaptic strengths as in Hennequin et al., Neuron, 2014.
    w0 = np.sqrt(2) * R / (np.sqrt(p * (1 - p) * (1 + gamma ** 2)))
    
    W = W * (w0 / np.sqrt(N)) # Excitatory synapses
    W[:, N//2:] = -gamma * W[:, N//2:]  # Inhibitory synapses
    
    return W