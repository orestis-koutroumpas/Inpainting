import scipy.io
import numpy as np
import matplotlib.pyplot as plt  

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Sigmoid Activation Function
def sigmoid(x):
    return 1. / (1. + np.exp(x))

# ReLU Derivative Function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Sigmoid Derivative Function
def sigmoid_derivative(x):
    return - np.exp(x) / (1 + np.exp(x))**2

# Cost Function
def J(N, norm, Z):
    return N * np.log(norm ** 2) + np.linalg.norm(Z) ** 2

# Plot learning curve
def plot_curves(cost_history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create 2x2 grid
    axes = axes.ravel()  # Flatten axes for easier indexing

    for i, cost_history in enumerate(costs_history):
        axes[i].plot(range(len(cost_history)), cost_history, label=f"Image {i+1}")
        axes[i].set_xlabel("Iterations")
        axes[i].set_ylabel("Cost")
        axes[i].set_title(f"Cost over Iterations for Image {i+1}")
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()  # Adjust layout to avoid overlapping
    plt.show()

# Fully Connected Neural Network for inpainting
class NeuralNetwork():
    def __init__(self, A1, A2, B1, B2, T, N):
        self.A1 = A1
        self.A2 = A2
        self.B1 = B1
        self.B2 = B2
        self.T = T
        self.N = N


    def forward(self, Z):
        # W1 = A1 ∗ Z + B1
        self.W1 = np.dot(self.A1, Z) + self.B1
        # Z1 = max{W1, 0} (ReLU)
        self.Z1 = relu(self.W1)
        # W2 = A2 ∗ Z1 + B2
        self.W2 = np.dot(self.A2, self.Z1) + self.B2
        # X = 1./[1 + exp(W2)](Sigmoid)
        self.X = sigmoid(self.W2)
        return self.X

    # Gradient Descent Algorithm with Learning Curve Plotting
    def gradient_descent(self, Z, Xn, step_size=0.0001, iterations=10000):
        cost_history = []
        Xn = Xn[:, np.newaxis] # (N,) -> (N,1)
        for iteration in range(iterations):
            # Calculate output X
            X = self.forward(Z)
            TX = np.dot(self.T, X)
            norm = np.linalg.norm(TX - Xn) # || TX - Xn ||
            cost = J(self.N, norm, Z) # Compute cost
            cost_history.append(cost)
            
            # Compute gradient
            grad_J = self.gradient(Z, TX, Xn, norm)
            
            # Update Z using gradient descent
            Z -= step_size * grad_J
                      
        return Z, cost_history

    
    def gradient(self, Z, TX, Xn, norm):  
        # u2 = ∇xφ(X), φ(X) = log( ||TX - XN||^2 ) 
        u2 = (2 / norm**2) * np.dot(self.T.T, (TX - Xn)) # u2 size 784 × 1
        # v2 = u2 ⋅ f2'(W2)
        v2 = u2 * sigmoid_derivative(self.W2) # v2 size 784 × 1
        # u1 = A2^T v2
        u1 = np.dot(self.A2.T, v2) # u1 size 128 × 1
        # v1 = u1 ⋅ f1'(W1)
        v1 = u1 * relu_derivative(self.W1) # v1 size 128 × 1
        # u0 = A1^T v1
        u0 = np.dot(self.A1.T, v1) # u0 size 10 × 1
        # ∇zJ(Z) = N uo + 2 Z
        gradJ = self.N * u0 + 2 * Z # ∇zJ(Z) size 10 × 1
        return gradJ

# Main Code
if __name__ == "__main__":
    # Load the generative model
    mat_data = scipy.io.loadmat('data/data1.mat')
    A1 = mat_data['A_1']  # Matrix A1 128 × 10
    A2 = mat_data['A_2']  # Matrix A2 784 × 128
    B1 = mat_data['B_1']  # Vector B1 128 × 1
    B2 = mat_data['B_2']  # Vector B2 784 × 1

    # Load Xi and Xn
    mat_data = scipy.io.loadmat('data/data2.mat')
    Xi = mat_data['X_i']  # Matrix Xi 784 × 4
    Xn = mat_data['X_n']  # Matrix Xn 784 × 4

    N = 400
    I = np.identity(N)  # N × N
    O = np.zeros((N, 784 - N))  # N × (784 - N)
    T = np.hstack((I, O))  # T = [I 0] size N × 784

    # Initialize Neural Network
    nn = NeuralNetwork(A1, A2, B1, B2, T, N)

    Z_optimal = []
    costs_history = []

    for i in range(4):
        print(f"\n\t\tProcessing Image {i+1}...")
        print(50*"=")
        Xn_i = Xn[:, i]  # Select each column of Xn
        X_N = np.dot(T, Xn_i)  # Isolate first N elements of Xn
        best_Z = None
        best_cost = float('inf')
        best_cost_history = []

        for j in range(20):
            Z = np.random.normal(0, 1, (10, 1))  # Input Z ~ N(0,1) size 10 × 1 
            Z_final, cost_history = nn.gradient_descent(Z, X_N)
            final_cost = cost_history[-1]

            # Keep track of the best Z based on final cost
            if final_cost < best_cost:
                best_cost = final_cost
                best_Z = Z_final
                best_cost_history = cost_history
            print(f"Initialization {j+1}: Minmum Value of J(Z) = {best_cost:4.5f}.")

        print(f"\nOptimum Cost for Image {i+1} is J(Z) = {best_cost:4.5f}.")
        Z_optimal.append(best_Z)
        costs_history.append(best_cost_history)

    # Plot the learning curves for the optimal solutions
    plot_curves(costs_history)
    
    # Create a 4x3 grid
    fig, axes = plt.subplots(4, 3, figsize=(12, 8))

    for i in range(4):
        # Ideal Xi image
        Xi_image = Xi[:, i].reshape(28, 28).T
        axes[i, 0].imshow(Xi_image, cmap='gray', origin='upper')
        axes[i, 0].set_title(f"Ideal Image {i+1}")
        axes[i, 0].axis('off')

        # Noise image
        Xn_i = Xn[:, i]  # Select each column
        XN = np.dot(T, Xn_i)  # Isolate first N elements of the image
        X_n = np.zeros((784, 1))  # Initialize zeros vector of size 784 × 1
        X_n[:N, 0] = XN  # Fill first N rows
        X_2D = X_n.reshape(28, 28).T  # Reshape & Transpose
        axes[i, 1].imshow(X_2D, cmap='gray', origin='upper')
        axes[i, 1].set_title(f"Isolated {N} pixels from Image {i+1} with Noise")
        axes[i, 1].axis('off')

        # Reconstructed Image
        nn_output = nn.forward(Z_optimal[i]).reshape(28, 28).T  # Neural network output
        axes[i, 2].imshow(nn_output, cmap='gray', origin='upper')
        axes[i, 2].set_title(f"Reconstructed Image {i+1}")
        axes[i, 2].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()