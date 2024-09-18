import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.linalg import det
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import svds

def get_edges_per_node(graph):
    """
    Get outgoing and incoming edges for each node in the graph.

    Args:
        graph (networkx.DiGraph): Directed graph.

    Returns:
        dict: A dictionary containing outgoing edges for each node.
        dict: A dictionary containing incoming edges for each node.
    """
    node_edges_out = {}
    node_edges_in = {}
    
    for node in graph.nodes:
        outgoing_edges = set(graph.successors(node))
        incoming_edges = set(graph.predecessors(node))
        
        edges_out = {(node, dst) for dst in outgoing_edges}
        edges_in = {(src, node) for src in incoming_edges}
        
        node_edges_out[node] = edges_out
        node_edges_in[node] = edges_in
        
    return node_edges_in, node_edges_out

def generate_matrix_M(k,
                      graph, 
                      node_edges_in, 
                      node_edges_out):
    """
    Generate a sparse matrix M_k based on the provided graph and parameters.

    Parameters:
        k (float): Wavenumber.
        graph (networkx.DiGraph): Directed graph.
        node_edges_in (dict): A dictionary containing incoming edges for each node.
        node_edges_out (dict): A dictionary containing incoming edges for each node.

    Returns:
        csr_matrix: Compressed Sparse Row matrix representing M_k.
    """
    num_edges = graph.number_of_edges()

    edge_to_index = {edge: idx for idx, edge in enumerate(graph.edges())}
    
    row = []
    col = []
    data = []

    idx = 0
    
    # Loop through nodes in the graph
    for node in graph.nodes:
        kirchhoff_row = []
        kirchhoff_col = []
        kirchhoff_data = []
        
        # continuity conditions
        if any(node_edges_in[node]):
            s_, t_ = list(node_edges_in[node])[0]
            i = edge_to_index[s_,t_]
            x_i = graph[s_][t_]['length']
            sin_i = np.sin(x_i*k)
            cos_i = np.cos(x_i*k)
            
            kirchhoff_col.extend([2*i, 2*i+1])
            kirchhoff_data.extend([-cos_i, sin_i])
            
            for s, t in list(node_edges_in[node])[1:]:
               
                j = edge_to_index[s,t]
                x_j = graph[s][t]['length']
                sin_j = np.sin(x_j*k)
                cos_j = np.cos(x_j*k)
                
                row.extend([idx, idx, idx, idx])
                col.extend([2*i, 2*i+1, 2*j, 2*j+1])
                data.extend([sin_i, cos_i, -sin_j, -cos_j])
            
                kirchhoff_col.extend([2*j, 2*j+1])
                kirchhoff_data.extend([-cos_j, sin_j])
                
                idx += 1
            
            for s, t in list(node_edges_out[node]):
                
                j = edge_to_index[s,t]
                x_j = 0 # graph[s_][t_]['length']
                sin_j = 0 # np.sin(x_j*k)
                cos_j = 1# np.cos(x_j*k)
                
                row.extend([idx, idx, idx, idx])
                col.extend([2*i, 2*i+1, 2*j, 2*j+1])
                data.extend([sin_i, cos_i, -sin_j, -cos_j])      
                
                kirchhoff_col.extend([2*j, 2*j+1])
                kirchhoff_data.extend([cos_j, -sin_j])
            
                idx += 1
                
        else:
            s_, t_ = list(node_edges_out[node])[0]
            
            i = edge_to_index[s_,t_]
            x_i = 0 # graph[s_][t_]['length']
            sin_i = 0 # np.sin(x_i*k)
            cos_i = 1# np.cos(x_i*k)

            kirchhoff_col.extend([2*i, 2*i+1])
            kirchhoff_data.extend([cos_i, -sin_i])
            
            for s, t in list(node_edges_out[node])[1:]:
                                
                j = edge_to_index[s,t]
                x_j = 0 # graph[s_][t_]['length']
                sin_j = 0 # np.sin(x_j*k)
                cos_j = 1# np.cos(x_j*k)
                
                row.extend([idx, idx, idx, idx])
                col.extend([2*i, 2*i+1, 2*j, 2*j+1])
                data.extend([sin_i, cos_i, -sin_j, -cos_j]) 
        
                kirchhoff_col.extend([2*j, 2*j+1])
                kirchhoff_data.extend([cos_j, -sin_j])
                
                idx += 1
                
        # Kirchhoff conditions
        row.extend(2*[idx]*(len(node_edges_in[node])+len(node_edges_out[node])))
        col.extend(kirchhoff_col)
        data.extend(kirchhoff_data)
        
        idx += 1
            
    matrix_M = csr_matrix((data, (row, col)), shape=(2*num_edges, 2*num_edges))

    return matrix_M
    
def inv_condition_number_sparse(k, 
                                graph, 
                                node_edges_in, 
                                node_edges_out,
                                return_M_k=False,
                                solver='arpack',
                                maxiter=False):
    """
    Calculate the inverse condition number of a sparse matrix M_k.

    Parameters:
        k (float): Parameter value.
        graph (networkx.DiGraph): Directed graph.
        node_edges_in (dict): A dictionary containing incoming edges for each node.
        node_edges_out (dict): A dictionary containing incoming edges for each node.
        solver: The solver used. ‘arpack’, ‘lobpcg’, and ‘propack’ are supported. Default: ‘arpack’.
        maxiter (int): Maximum number of Arnoldi update iterations allowed.

    Returns:
        float: Inverse condition number of the matrix M_k.
    """
    k = float(k)
    M_k = generate_matrix_M(k, graph, node_edges_in, node_edges_out)

    # Calculate the singular values using SVDs
    if maxiter == False:
        max_singular_value = svds(M_k, k=1, which='LM', \
        return_singular_vectors=False, solver=solver)
        min_singular_value = svds(M_k, k=1, which='SM', \
        return_singular_vectors=False, solver=solver)
    else:
        max_singular_value = svds(M_k, k=1, which='LM', \
        return_singular_vectors=False, solver=solver, maxiter=maxiter)
        min_singular_value = svds(M_k, k=1, which='SM', \
        return_singular_vectors=False, solver=solver, maxiter=maxiter)
            
    # Compute the condition number as the ratio of the largest to smallest singular value
    inv_condition_number = min_singular_value / max_singular_value
    
    if return_M_k:
        return inv_condition_number, M_k
    else:
        return inv_condition_number

def analyze_k_values(k_values,
                     graph, 
                     node_edges_in, 
                     node_edges_out,
                     threshold_inv_condition_number, 
                     return_det=False,
                     solver='arpack',
                     maxiter=False):
    """
    Analyze a range of wavenumber values for a given network.

    Parameters:
        k_values (numpy.ndarray): Array of wavenumber values to analyze.
        graph (networkx.DiGraph): Directed graph.
        node_edges_in (dict): A dictionary containing incoming edges for each node.
        node_edges_out (dict): A dictionary containing incoming edges for each node.
        threshold_inv_condition_number (float): Threshold for inverse condition number.
        return_det (bool, optional): Flag to determine whether to return determinants (based on dense matrices).
        solver: The solver used. ‘arpack’, ‘lobpcg’, and ‘propack’ are supported. Default: ‘arpack’.
        maxiter (int): Maximum number of Arnoldi update iterations allowed.
        
    Returns:
        list: List of singular points.
        list: List of inverse condition numbers.
        list (optional): List of determinants (if return_det is True).
    """
    singular_points = []
    inv_condition_numbers = []
    det_vals = []

    for k in k_values:
        inv_condition_number, M_k = inv_condition_number_sparse(k, graph, node_edges_in, node_edges_out, \
                                                                return_M_k=True, solver=solver, maxiter=maxiter)
                                                                
        if return_det:
            det_vals.append(det(M_k.todense()))

        inv_condition_numbers.append(inv_condition_number)

        if (inv_condition_number[0] < threshold_inv_condition_number) or (inv_condition_number[0] == 0):
            singular_points.append(k)

    if return_det:
        return singular_points, inv_condition_numbers, det_vals
    else:
        return singular_points, inv_condition_numbers
        
def find_unique_optimal_k(singular_points, 
                          delta_k, 
                          graph, 
                          node_edges_in, 
                          node_edges_out, 
                          round_dec,
                          xatol=1e-15,
                          solver='arpack',
                          maxiter=False):
    """
    Find unique optimal k values within a specified range.

    Parameters:
        singular_points (list): List of singular points.
        delta_k (float): Range for optimization.
        graph (networkx.DiGraph): Directed graph.
        node_edges_in (dict): A dictionary containing incoming edges for each node.
        node_edges_out (dict): A dictionary containing incoming edges for each node.
        round_dec (int): Number of decimal places to round the results.
        xatol (float): Tolerance of the minimize_scalar optimizer.
        solver: The solver used. ‘arpack’, ‘lobpcg’, and ‘propack’ are supported. Default: ‘arpack’.
        maxiter (int): Maximum number of Arnoldi update iterations allowed.
                
    Returns:
        numpy.ndarray: Array of unique rounded optimal k values.
    """
    k_opt_vals = []

    for k_ in singular_points[1:]:
        result = minimize_scalar(inv_condition_number_sparse, args=(graph, node_edges_in, node_edges_out, False, solver, maxiter), \
                                 bounds=(k_-delta_k, k_+delta_k), method='bounded', options={'xatol': xatol})
        optimal_k = result.x
        k_opt_vals.append(float(optimal_k))

    rounded_elements = [round(element, round_dec) for element in k_opt_vals]
    k_opt_vals_unique = np.array(list(set(rounded_elements)))

    return k_opt_vals_unique
    
def calculate_k_multiplicity(k_opt_vals_unique, 
                             graph, 
                             node_edges_in, 
                             node_edges_out, 
                             tolerance,
                             return_eig_vals=False):
    """
    Calculate the unique k values with multiplicity.

    Parameters:
        k_values (numpy.ndarray): Array of unique rounded optimal k values.
        graph (networkx.DiGraph): Directed graph.
        node_edges_in (dict): A dictionary containing incoming edges for each node.
        node_edges_out (dict): A dictionary containing incoming edges for each node.
        tolerance (float): Tolerance for zero singular values.
        return_eig_vals (bool, optional): Flag to determine whether to return eigenvectors.

    Returns:
        numpy.ndarray: Array of unique k values with multiplicity.
        
    Notes:
    One may use numpy.linalg.svd for dense matrices:

    U, S, VT = np.linalg.svd(M_k)

    The scipy.sparse svds is problematic:

    U, S, VT = svds(M_k, k=1, which='SM')

    - https://github.com/scipy/scipy/issues/11406
    - https://github.com/scipy/scipy/issues/3452
    - https://stackoverflow.com/questions/33410146/how-can-i-compute-the-null-space-kernel-x-m-x-0-of-a-sparse-matrix-in-python

    Here we use both https://github.com/yig/PySPQR and numpy to compute the null space of M_k.
    """
    k_opt_vals_unique_multiplicity = []
    eig_vals = []
    
    for k_m in k_opt_vals_unique:
        M_k = generate_matrix_M(k_m, graph, node_edges_in, node_edges_out)
        #sM_k = csr_matrix(M_k)
        # Perform sparse SVD on M_k with k singular values
        u, s, vt = svds(M_k.T, k=min(M_k.shape)-1, which="SM")

        # Identify the indices of singular values that are close to zero
        zero_singular_indices = np.where(np.isclose(s, 0, atol=tolerance))[0]

        # Extend the list with the current value of k_m based on zero singular values' multiplicity
        k_opt_vals_unique_multiplicity.extend(len(zero_singular_indices) * [k_m])

        # For each index of zero singular values, extract the corresponding row of V^T
        for i in zero_singular_indices:
            VT = vt[i, :]
            eig_vals.append(VT)
    
    k_opt_vals_unique_multiplicity = np.array(k_opt_vals_unique_multiplicity)
    
    if return_eig_vals:
        return k_opt_vals_unique_multiplicity, eig_vals
    else:
        return k_opt_vals_unique_multiplicity
