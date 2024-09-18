import numpy as np
import streamlit as st
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from utilities.network_generators import star_graph
from utilities.sound import *

# customized settings
params = {  # 'backend': 'ps',
    'font.size': 8,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'text.usetex': False}
# tell matplotlib about your params
rcParams.update(params)

# Title for the Streamlit app
st.title("The Sound of a Metric Star Network")

# Inputs for edge lengths
st.subheader("Set Edge Lengths")
edge_length_0 = st.number_input("Length of edge 0-3", min_value=0.1, value=1.0, step=0.1)
edge_length_1 = st.number_input("Length of edge 1-3", min_value=0.1, value=1.0, step=0.1)
edge_length_2 = st.number_input("Length of edge 2-3", min_value=0.1, value=1.0, step=0.1)

# Create a list of edge lengths
edge_lengths = [edge_length_0, edge_length_1, edge_length_2]

st.subheader("Parameters for Computing Characteristic Wavenumbers")

num_k = st.number_input("Number of samples to generate", min_value=100, value=1000, step=100)
threshold_inv_condition_number = st.number_input("Inversion condition number threshold", min_value=1e-6, value=1e-2, step=1e-3, format="%.6f")
delta_k = st.number_input("Bracketing value", min_value=1e-6, value=0.1, step=0.1, format="%.6f")
round_dec = st.number_input("Decimals to round characteristic-wavenumber candidates", min_value=1, value=6, step=1)
tolerance = st.number_input("Tolerance for zero singular values", min_value=1e-9, value=1e-6, step=1e-6, format="%.9f")

# Generate the directed star graph and analyze k-values
G, k_values, inv_condition_numbers, \
k_opt_vals_unique, k_opt_vals_unique_multiplicity = star_graph(edge_lengths, num_k,\
                                                               threshold_inv_condition_number, delta_k, \
                                                               round_dec, tolerance)

# Visualize the graph

# Define the custom positions
# Center node at (0, 0)
positions = {3: (0, 0)}

# Outer nodes at 120° intervals, scaled by edge lengths
angles = [0, 120, 240]  # Angles in degrees

for i, angle in enumerate(angles):
    # Convert angle to radians
    angle_rad = np.radians(angle)
    # Calculate the x, y coordinates using the edge lengths
    x = 0.02*edge_lengths[i] * np.cos(angle_rad)
    y = 0.02*edge_lengths[i] * np.sin(angle_rad)
    positions[i] = (x, y)

node_color_rgba = [mcolors.to_rgba("tab:blue", alpha=0.7) \
                   for _ in range(G.number_of_nodes())]
edge_color_rgba = mcolors.to_rgba("LightGrey", alpha=0.9)
node_edge_color_rgba = mcolors.to_rgba("tab:blue", alpha=0.9)

st.subheader("Network Visualization")
fig, ax = plt.subplots(figsize=(10, 10))
nx.draw(G, pos=positions, with_labels=False, node_color=node_color_rgba,
        edge_color=edge_color_rgba, node_size=800, font_size=10, ax=ax,
        edgecolors=node_edge_color_rgba)
st.pyplot(fig)

# Display the unique optimal k values
st.subheader("Characteristic Wavenumbers")
st.markdown(
    r"""
    The left panel shows the inverse condition number $\kappa(T(k))^{-1}$ as a function of the characteristic wavenumber $k$. 
    The right panel shows the characteristic-wavenumber counting function $N_\mathcal{G}(k)$ (solid black curve) as a function of $k$ and Weyl's law 
    $N_\mathcal{G}(k) = \frac{k \mathcal{L}}{\pi}$ (dashed black curve), where $\mathcal{L}$ is the total length of the metric network. 
    In this plot, we include the "zero mode" (for which $k_m = 0$) in $N_\mathcal{G}(k)$. 
    The dash-dotted and dash-dot-dotted gray lines, respectively, indicate the lower and upper bounds of Weyl's law.
    """
)

fig, axs = plt.subplots(1,2, figsize=(5, 2.5))

axs[0].plot(k_values,inv_condition_numbers,lw=1,color='k')
axs[0].set_xlim(0,20)
axs[0].set_ylim(0,0.4)
axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
axs[0].set_xlabel(r"$k$")
axs[0].set_ylabel(r"$\kappa(T(k))^{-1}$")

axs[1].plot(k_values, [len(k_opt_vals_unique_multiplicity[k_opt_vals_unique_multiplicity <= k]) for k in k_values],lw=1,color="k")
axs[1].plot(k_values, 3/np.pi*k_values,lw=1,color="k",ls=(0, (5, 1)),label=r"$\frac{k\mathcal{L}}{\pi}$")
axs[1].plot(k_values, 3/np.pi*k_values+4,lw=1,color="Gray",ls=(0, (3, 1, 1, 1, 1, 1)),label=r"$\frac{k\mathcal{L}}{\pi}+N$")
axs[1].plot(k_values, 3/np.pi*k_values-3,lw=1,color="Gray",ls=(0, (3, 1, 1, 1)),label=r"$\frac{k\mathcal{L}}{\pi}-M$")
axs[1].set_xlim(0,20)
axs[1].set_ylim(0,40)
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(2))
axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
axs[1].yaxis.set_major_locator(ticker.MultipleLocator(10))
axs[1].set_xlabel(r"$k$")
axs[1].set_ylabel(r"$N_{\mathcal{G}}(k)$")
axs[1].legend(loc=0,frameon=False,handlelength=1.4)

plt.tight_layout()
st.pyplot(fig)

st.subheader("Making Wave Numbers Audible")

st.markdown(
    r"""
    Given a phase velocity $v_{\mathrm{p}}=\omega/k$, the corresponding frequency is $f=k\, v_{\mathrm{p}}/(2 \pi)$.
    The sound file below makes the first few characteristic wavenumbers audible.
    """
)

# Generate tones for all wave numbers and concatenate them into one audio file
all_tones = []

# Duration for each tone
duration = 2.0

# Loop over each wave number and generate its corresponding tone
for k in k_opt_vals_unique[k_opt_vals_unique != 0]:

    # Convert the k value to a frequency
    frequency = wave_number_to_frequency(k)
    
    # Scale frequency to the audible range (optional)
    if frequency < 20:
        frequency *= 100
    elif frequency > 20000:
        frequency /= 10
    
    # Generate the tone for the current frequency
    tone = generate_tone(frequency, duration)
    
    # Append the generated tone to the list
    all_tones.append(tone)

# Concatenate all tones into a single array
combined_tone = np.concatenate(all_tones)

# Convert the generated tone to a WAV file format
wav_file = create_wav_file(combined_tone)

# Use st.audio to play the tone
st.audio(wav_file, format="audio/wav")
