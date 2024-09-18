import numpy as np
import networkx as nx
from utilities.utilities_spectral import *

def star_graph(edge_lengths, num_k=1000, threshold_inv_condition_number=1e-2, \
               delta_k=0.1, round_dec=6, tolerance=1e-6):
    """
    Generates a directed star graph with customizable edge lengths and performs analysis on it.

    Parameters:
    - edge_lengths: List of lengths for the edges in the star graph.

    Returns:
    - G: The directed graph.
    - k_opt_vals_unique: List of unique optimal k values.
    """
    # Create the directed graph
    G = nx.DiGraph()

    # Ensure edge_lengths is a list of the correct length (3 edges)
    if len(edge_lengths) != 3:
        raise ValueError("edge_lengths must have exactly 3 values.")

    # Add edges with the specified lengths
    G.add_edge(0, 3, length=edge_lengths[0])
    G.add_edge(1, 3, length=edge_lengths[1])
    G.add_edge(2, 3, length=edge_lengths[2])

    # Get unique edge pairs per node
    node_edges_in, node_edges_out = get_edges_per_node(G)

    # Define the range of k values to analyze
    k_values = np.linspace(0, 20, num_k)

    # Analyze k values
    singular_points, inv_condition_numbers, det_vals = analyze_k_values(
        k_values, G, node_edges_in, node_edges_out, threshold_inv_condition_number, return_det=True
    )  

    # Find unique optimal k values
    k_opt_vals_unique = find_unique_optimal_k(
        singular_points, delta_k, G, node_edges_in, node_edges_out, round_dec
    )

    k_opt_vals_unique_multiplicity = calculate_k_multiplicity(k_opt_vals_unique, G, \
                                                              node_edges_in, node_edges_out, \
                                                              tolerance)
    
    k_opt_vals_unique_multiplicity = np.insert(k_opt_vals_unique_multiplicity,0,0)
                                                              
    return G, k_values, inv_condition_numbers, \
    k_opt_vals_unique, k_opt_vals_unique_multiplicity
