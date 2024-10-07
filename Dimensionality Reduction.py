import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def k_nearest_neighbors_graph(data, k):
    n = len(data)
    graph = {i: [] for i in range(n)}
    
    for i in range(n):
        distances = [(euclidean_distance(data[i], data[j]), j) for j in range(n) if i != j]
        distances.sort()
        
        for d, j in distances[:k]:
            graph[i].append((j, d))
    
    return graph


def compute_gradient(Y, shortest_paths):
    grad = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[0]):
            if i == j:
                continue
            pi = shortest_paths[(i, j)]
            if pi == float('inf'): 
                continue
            Y_dist = np.linalg.norm(Y[i] - Y[j])
            pair_grad = 2 * (Y[i] - Y[j]) - 2 * pi * (Y[i] - Y[j]) / (Y_dist + 1e-10)
            
            grad[i] += pair_grad

    return grad

def compute_stochastic_gradient(X, Y, graph, shortest_paths, i):
    grad = np.zeros_like(Y[i])
    for j in range(Y.shape[0]):
        if i == j:
            continue
        pi = shortest_paths[(i, j)]
        if pi == float('inf'):
            continue
        Y_dist = np.linalg.norm(Y[i] - Y[j])
        pair_grad = 2 * (Y[i] - Y[j]) - 2 * pi * (Y[i] - Y[j]) / (Y_dist + 1e-10)
        
        grad += pair_grad
    return grad


def gradient_descent(X, shortest_paths, d, learning_rate=0.01, max_iterations=200):
    Y = np.random.rand(X.shape[0], d)
    curr_Y = Y.copy()

    for t in range(max_iterations):
        grad = compute_gradient(Y, shortest_paths)
        Y = Y - learning_rate * grad

        if np.max(np.abs(Y - curr_Y)) < 1e-6:
            print("Convergence achieved.")
            return Y
        
        curr_Y = Y.copy()
        
        print(f"Iteration {t}")

    return Y

def stochastic_gradient_descent(X, graph, shortest_paths, d, learning_rate=0.01, max_iterations=200):
    Y = np.random.rand(X.shape[0], d)
    curr_Y = Y.copy()
    
    for t in range(max_iterations):
        print(f"Iteration {t}")
        shuffled_indices = np.random.permutation(X.shape[0])
        
        for i in shuffled_indices:
            grad = compute_stochastic_gradient(X[i], Y, graph, shortest_paths, i)

            Y[i] = Y[i] - learning_rate * grad
            
        if np.max(np.abs(Y - curr_Y)) < 1e-6:
            print("Convergence achieved.")
            return Y
            
        curr_Y = Y.copy()

    return Y


def all_pairs_shortest_path(graph):
    n = len(graph)

    shortest_paths = np.full((n, n), np.inf)

    np.fill_diagonal(shortest_paths, 0)

    for i in graph:
        for j, weight in graph[i]:
            shortest_paths[i, j] = weight

    for k in range(n):
        shortest_paths = np.minimum(shortest_paths, shortest_paths[:, k, None] + shortest_paths[None, k, :])

    return shortest_paths


def simulated_annealing(objective, X, d, n_iterations, step_size, temp, shortest_paths):
    Y = np.random.rand(X.shape[0], d)
    best_eval = objective(Y, shortest_paths)
    curr_Y, curr_eval = Y, best_eval
    for i in range(n_iterations):
        candidate_Y = curr_Y + np.random.randn(X.shape[0], d) * step_size
        candidate_eval = objective(candidate_Y, shortest_paths)

        if candidate_eval < best_eval:
            curr_Y, best_eval = candidate_Y, candidate_eval

        diff = candidate_eval - curr_eval
        t = temp / float(i + 1)
        prob = np.exp(-diff / t)
        
        if diff < 0 or np.random.rand() < prob:
            curr_Y, curr_eval = candidate_Y, candidate_eval
    return [curr_Y, best_eval]


def objective(Y, shortest_paths):
    n = Y.shape[0]
    loss = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pi = shortest_paths[(i, j)]
                if pi == float('inf'): 
                    continue
                loss += (np.linalg.norm(Y[i] - Y[j]) - pi)**2
    return loss


def pca(X, d):
    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    covariance_matrix = np.cov(X_standardized.T)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvectors = eigenvectors[:, :d]

    X_pca = np.dot(X_standardized, eigenvectors)

    return X_pca


def main():
    swiss_roll = np.loadtxt('swiss_roll.txt')
    
    x = swiss_roll[:, 0]
    y = swiss_roll[:, 1]
    z = swiss_roll[:, 2]

    s = np.sqrt(x**2 + y**2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis((s - s.min()) / (s.max() - s.min()))
    scatter = ax.scatter(x, y, z, c=colors)
    cbar = plt.colorbar(scatter)

    plt.show()
    
    
    #PCA
    X_pca = pca(swiss_roll, num_components=2)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c = colors, alpha=0.8, lw=2)
    plt.title('PCA of Swiss Roll')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()
    
    
    #2D Embedding
    graph = k_nearest_neighbors_graph(swiss_roll, 3)
    print("Done with Graph")
    shortest_paths = all_pairs_shortest_path(graph)
    print("done with paths")
    
    
    #Gradient Descent
    X_new = gradient_descent(swiss_roll, shortest_paths, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_new[:, 0], X_new[:, 1], c = colors, alpha=0.8, lw=2)
    plt.title('2D Embedding of Swiss Roll')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.show()


    #Simulated Annealing
    X_new, score = simulated_annealing(objective, swiss_roll, 2, 1000, 0.1, 10, shortest_paths)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_new[:, 0], X_new[:, 1], c = colors, alpha=0.8, lw=2)
    plt.title('2D Embedding of Swiss Roll')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.show()


    swiss_roll_hole = np.loadtxt('swiss_roll_hole.txt')
    
    x = swiss_roll_hole[:, 0]
    y = swiss_roll_hole[:, 1]
    z = swiss_roll_hole[:, 2]

    s = np.sqrt(x**2 + y**2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis((s - s.min()) / (s.max() - s.min()))
    scatter = ax.scatter(x, y, z, c=colors)
    cbar = plt.colorbar(scatter)

    plt.show()
    
    
    #PCA
    X_pca = pca(swiss_roll_hole, num_components=2)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c = colors, alpha=0.8, lw=2)
    plt.title('PCA of Swiss Roll')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()

    
    #2D Embedding
    graph = k_nearest_neighbors_graph(swiss_roll, 3)
    print("Done with Graph")
    shortest_paths = all_pairs_shortest_path(graph)
    print("done with paths")
    
    
    #Gradient Descent
    X_new = gradient_descent(swiss_roll_hole, shortest_paths, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_new[:, 0], X_new[:, 1], c = colors, alpha=0.8, lw=2)
    plt.title('2D Embedding of Swiss Roll')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.show()
    
    
    #Simulated Annealing
    X_new, score = simulated_annealing(objective, swiss_roll_hole, 2, 1000, 0.1, 10, shortest_paths)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_new[:, 0], X_new[:, 1], c = colors, alpha=0.8, lw=2)
    plt.title('2D Embedding of Swiss Roll')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.show()
    

if __name__ == "__main__":
    main()


