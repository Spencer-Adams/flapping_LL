import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.animation as animation  # type: ignore
import os
import flappingSetup as example_setup # type: ignore
from flappingSetup import Ra, bw, cw, Rt, CLa, aoa_deg, Omega_deg, das_deg, pbar, sar, sat, fcf_ar, fcf_at, de, he, car, cat, har, hat, isElliptic
from tqdm import tqdm # type: ignore
N_nodes = 99 # number of nodes for the lifting line
# nomenclature:
# Ra = aspect ratio
# bw = wing span
# cw = mean chord
# Rt = taper ratio
# CLa = section lift slope
# aoa_deg = angle of attack at the root in degrees (alpha-alpha_LO)root
# Omega_deg = maximum washout amount in degrees
# das_deg = delta anti-symmetric deflection (aileron) in degrees
# pbar = dimensionless roll rate
# sar = spanwise location of aileron at root
# sat = spanwise location of aileron at tip
# fcf_ar = flap chord fraction at root aileron
# fcf_at = flap chord fraction at tip aileron
# de = deflection efficiency (aileron)
# he = hinge efficiency (aileron)
# car = chord at root aileron
# cat = chord at tip aileron
# har = hinge location at root aileron
# hat = hinge location at tip aileron
# theta = spanwise location of the node
# z/b = spanwise location of the node
# chord = chord distribution
# washout = washout distribution
# Chi = aileron distribution
# C = matrix for the lifting line
# C_inv = inverse of the C matrix 
# a = Fourier coefficient for alpha
# b = Fourier coefficient for washout distribution
# c = Fourier coefficient for aileron distribution
# d = Fourier coefficient for roll rate 

def calculate_thetas():
    """This function calculates the theta values for the lifting line"""
    theta_array = np.linspace(0, np.pi, N_nodes)
    return theta_array

def calculate_zb(theta_array):
    """This function calculates the z/b values for the lifting line, zb is the spanwise location of the node"""
    z_b_array = np.zeros((N_nodes))
    z_b_array[0] = 0.5
    z_b_array[N_nodes - 1] = -0.5
    for i in range(1, N_nodes -1):
        z_b_array[i] = (1/2)*np.cos(theta_array[i])
    return z_b_array

def calc_chord_dist(theta_array):
    """Takes in the theta array and returns the chord distribution array"""
    chord_distribution_array = np.zeros((N_nodes))
    for i in range(0, N_nodes):
        chord_distribution_array[i] = example_setup.chord(theta_array[i])
    return chord_distribution_array

def calc_washout_dist(theta_array):
    """This function calculates the washout distribution for the wing"""
    washout_distribution = np.zeros((N_nodes))
    for i in range(0, N_nodes):
        washout_distribution[i] = example_setup.omega(theta_array[i])
    return washout_distribution

def calc_Chi_dist(theta_array):
    chi_array = np.zeros((N_nodes))
    for i in range(0, N_nodes):
        chi_array[i] = example_setup.chi(theta_array[i])
    return chi_array    

def calc_C_matrix(N_nodes, theta_array, chord_distribution_array):
    """This function calculates the C matrix for the lifting line"""
    C_matrix = np.zeros((N_nodes, N_nodes))
    for i in range(N_nodes):
        for j in range(1, N_nodes + 1):
            if not isElliptic:
                if i == 0:
                    C_matrix[i, j-1] = j ** 2 # using l'Hopital's rule to find the limit as theta approaches 0
                elif i == N_nodes - 1:
                    C_matrix[i, j-1] = (-1)**(j+1)*j**2 # using l'Hopital's rule to find the limit as theta approaches pi
                else: 
                    C_matrix[i, j-1] = (((4 * bw) / (CLa * chord_distribution_array[i])) + (j) / np.sin(theta_array[i])) * np.sin((j)*theta_array[i])
            else:
                if i == 0:
                    C_matrix[i, j-1] = (np.pi*Ra/CLa + j)*j # using l'Hopital's rule to find the limit as theta approaches 0
                elif i == N_nodes - 1:
                    C_matrix[i, j-1] = (np.pi*Ra/CLa +j)*((-1)**(j+1)*j)# using l'Hopital's rule to find the limit as theta approaches pi
                else:
                    C_matrix[i, j-1] = (((4 * bw) / (CLa * chord_distribution_array[i])) + (j) / np.sin(theta_array[i])) * np.sin((j)*theta_array[i])
    return C_matrix

def a_fourier(a_fourier_coeff_vec): # these Lift values are acting at each node
    """
    Calculate the Fourier coefficient due to angle of attack.
    Parameters:
    a_fourier_coeff_vec (numpy.ndarray): A vector of Fourier coefficients.
    
    Returns:
    numpy.ndarray: The Total Fourier Coefficient due to angle of attack.
    """
    a = np.zeros((N_nodes))
    for j in range(N_nodes):
        a[j] = a_fourier_coeff_vec[j]*np.deg2rad(aoa_deg) # First term in eq 1.8.50 in mech of flight 
    return a

def b_fourier(b_fourier_coeff_vec):
    """Calculate the Fourier coefficient due to washout distribution.
    Parameters:
    b_fourier_coeff_vec (numpy.ndarray): A vector of Fourier coefficients.
    
    Returns:
    numpy.ndarray: The Total Fourier Coefficient due to washout distribution.
    """
    b = np.zeros((N_nodes))
    for j in range(N_nodes):
        b[j] = -b_fourier_coeff_vec[j]*np.deg2rad(Omega_deg) # Second term in eq 1.8.50 in mech of flight
    return b

def c_fourier(c_fourier_coeff_vec):
    """Calculate the Fourier coefficient due to aileron distribution.
    Parameters:
    c_fourier_coeff_vec (numpy.ndarray): A vector of Fourier coefficients.

    Returns:
    numpy.ndarray: The Total Fourier Coefficient due to aileron distribution.
    """
    C_from_aileron = np.zeros((N_nodes))
    for j in range(N_nodes):
        C_from_aileron[j] = c_fourier_coeff_vec[j]*np.deg2rad(das_deg) # Third term in eq 1.8.50 in mech of flight
    return C_from_aileron

def d_fourier(d_fourier_coeff_vec):
    """Calculate the Fourier coefficient due to roll rate.
    Parameters:
    d_fourier_coeff_vec (numpy.ndarray): A vector of Fourier coefficients.

    Returns:
    numpy.ndarray: The Total Fourier Coefficient due to roll rate.
    """
    D_from_roll_rate = np.zeros((N_nodes))
    for j in range(N_nodes):
        D_from_roll_rate[j] = d_fourier_coeff_vec[j]*pbar # Fourth term in eq 1.8.50 in mech of flight
    return D_from_roll_rate

def A_fourier_total(a_fourier, b_fourier, c_fourier, d_fourier):
    """Calculate the total lift distribution.
    Parameters:
    a_fourier (numpy.ndarray): The Total Fourier Coefficient due to angle of attack.
    b_fourier (numpy.ndarray): The Total Fourier Coefficient due to washout distribution.
    c_fourier (numpy.ndarray): The Total Fourier Coefficient due to aileron distribution.
    d_fourier (numpy.ndarray): The Total Fourier Coefficient due to roll rate.
    Returns:
    numpy.ndarray: The Total Fourier Coefficient due to all factors.
    """
    A_total = a_fourier + b_fourier + c_fourier + d_fourier
    A_symmetric = a_fourier + b_fourier
    A_asymmetric = c_fourier + d_fourier
    return A_total, A_symmetric, A_asymmetric

def calc_CL(Fourier_Coeff_A):
    """Calculate the lift coefficient.
    Parameters:
    Fourier_Coeff_A (numpy.ndarray): The fourier coefficient due to any one or all of the factors.

    Returns:
    float: The lift coefficient.
    """
    CL = np.pi * Ra * Fourier_Coeff_A[0] # CL = pi*RA*A1 -> eq 6.5 in aero eng handbook 
    return CL

def calc_CDi(Fourier_Coeff_A):
    """Calculate the induced drag coefficient.
    Parameters:
    Fourier_Coeff_A (numpy.ndarray): The fourier coefficient due to any one or all of the factors.

    Returns:
    float: The induced drag coefficient.
    """
    CDi = 0.0
    multiplier = np.pi * Ra 
    inside_term = 0.0
    for j in range(1, N_nodes + 1):
        if (j) % 2 != 0: # if j is odd
            e_n = (4*(-1)**((j+1)/2))/((j**2-4)*np.pi)
        else:
            e_n = 0
        inside_term += ((j)*Fourier_Coeff_A[j-1]**2 - Fourier_Coeff_A[j-1]*e_n*pbar)
    CDi = multiplier * (inside_term) # eq 4 in assignment description
    return CDi

def calc_Cl(Fourier_Coeff_A):
    """Calculate the rolling moment distribution.
    Parameters:
    Fourier_Coeff_A (numpy.ndarray): The fourier coefficient due to any one or all of the factors.

    Returns:
    numpy.ndarray: The lift distribution.
    """
    Cl = -(np.pi*Ra*Fourier_Coeff_A[1])/(4) # eq 5 in assignment description
    return Cl

def calc_Cn(Fourier_Coeff_A):
    """Calculate the normal force distribution.
    Parameters:
    Fourier_Coeff_A (numpy.ndarray): The fourier coefficient due to any one or all of the factors.

    Returns:
    numpy.ndarray: The normal force distribution.
    """
    Cn = 0.0
    for j in range(2, N_nodes + 1):
        Cn += ((2*(j)-1)*(Fourier_Coeff_A[j-2]*Fourier_Coeff_A[j-1]))
    Cn = (np.pi*Ra/4)*(-0.5*(Fourier_Coeff_A[0]+Fourier_Coeff_A[2])*pbar + Cn) # eq 6 in assignment description 
    return Cn

def calc_CL_distribution(Fourier_Coeff_A, theta_array):
    """Calculate the lift distribution.
    Parameters:
    Fourier_Coeff_A (numpy.ndarray): The fourier coefficient due to any one or all of the factors.
    theta_array (numpy.ndarray): The theta values.
    Returns:
    numpy.ndarray: The lift distribution.
    """
    CL_distribution = np.zeros((N_nodes))
    inverted_theta_array = np.flip(theta_array) # flip the theta array to get in correct coordinate system
    for i in range(N_nodes):
        for j in range(N_nodes):
            CL_distribution[i] += 4*Ra*(Fourier_Coeff_A[j] * np.sin((j+1)*inverted_theta_array[i])) # eq 1 in assignment description
    return CL_distribution

def calc_CD_distribution(Fourier_Coeff_A, theta_array):
    """Calculate the lift distribution.
    Parameters:
    Fourier_Coeff_A (numpy.ndarray): The fourier coefficient due to any one or all of the factors.
    theta_array (numpy.ndarray): The theta values.
    Returns:
    numpy.ndarray: The lift distribution.
    """
    CD_distribution = np.zeros((N_nodes))
    inverted_theta_array = np.flip(theta_array) # flip the theta array to get in correct coordinate system
    first = np.zeros((N_nodes))
    second = np.zeros((N_nodes))
    for i in range(N_nodes):
        for j in range(N_nodes):
            first[i] += 4*Ra*(Fourier_Coeff_A[j] * np.sin((j+1)*inverted_theta_array[i])) # eq 1 in assignment description
            second[i] += (Fourier_Coeff_A[j] * np.sin((j+1)*inverted_theta_array[i])) # eq 1 in assignment description
            third = abs(np.cos(inverted_theta_array[i])) * pbar 
            CD_distribution[i] += first[i] * (third - second[i]) # eq 1 in assignment description
    return CD_distribution

def calc_aileron_position():
    """Calculate the aileron position.
    Parameters:
    theta_array (numpy.ndarray): The theta values.
    
    Returns:
    numpy.ndarray: The aileron position.
    """
    trail_edge_at_sar = 0.75 * car # car = chord at root aileron
    trail_edge_at_sar = normalize_one_chord_by_span(trail_edge_at_sar)
    # calculate the triling edge at the tip
    trail_edge_at_sat = 0.75 * cat # cat = chord at tip aileron
    trail_edge_at_sat = normalize_one_chord_by_span(trail_edge_at_sat)
    harr = normalize_one_chord_by_span(har) # har = hinge location at root aileron
    hatt = normalize_one_chord_by_span(hat) # hat = hinge location at tip aileron
    # now assemble the x and y positions of the aileron
    aileron_position_right = np.array([[sar/2, trail_edge_at_sar], [sar/2, harr], [sat/2, hatt], [sat/2, trail_edge_at_sat]])
    # now mirror the aileron position to the left side
    aileron_position_left = np.array([[-sar/2, trail_edge_at_sar], [-sar/2, harr], [-sat/2, hatt], [-sat/2, trail_edge_at_sat]])
    return aileron_position_right, aileron_position_left

def normalize_chord_by_span(chord_distribution_array):
    """Normalize the chord and span by the maximum span.
    Parameters:
    chord_distribution_array (numpy.ndarray): The chord distribution values.
    z_b_array (numpy.ndarray): The z/b values.
    
    Returns:
    numpy.ndarray: The normalized chord distribution values.
    numpy.ndarray: The normalized z/b values.
    """
    normalized_chord_distribution_array = np.zeros((N_nodes))
    for i in range(0, N_nodes):
        normalized_chord_distribution_array[i] = normalize_one_chord_by_span(chord_distribution_array[i])
        
    return normalized_chord_distribution_array

def normalize_one_chord_by_span(chord):
    """Normalize one chord by the maximum span.
    Parameters:
    chord (float): The chord value.
    
    Returns:
    float: The normalized chord value.
    """
    return chord / bw

# def create_combined_distribution_figure(z_b_array, chord_distribution_array, CL_distributions, distribution_names, titles):
#     """
#     Create a combined figure with multiple subplots, adjusted for better spacing and proper aspect ratio.
#     Parameters:
#     z_b_array (numpy.ndarray): The z/b values.
#     chord_distribution_array (numpy.ndarray): The chord distribution values.
#     CL_distributions (list of numpy.ndarray): List of CL distributions to plot.
#     distribution_names (list of str): Names of the distributions.
#     titles (list of str): Titles for each subplot.
#     """
#     # Create a 4x2 grid for subplots
#     fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 12))
#     plt.subplots_adjust(hspace=1.2, wspace=0.4)  # Increase spacing between plots

#     # Function to calculate dynamic font size
#     def calculate_fontsize(title, ax_width):
#         max_fontsize = 16
#         min_fontsize = 12
#         title_length = len(title)
#         fontsize = min(max_fontsize, max(min_fontsize, ax_width / title_length * 1.5))
#         return fontsize

#     # Planform plot (top-left)
#     ax = axes[0, 0]
#     ax.grid()
#     normalized_chord_distribution_array = normalize_chord_by_span(chord_distribution_array)
#     leading_edge_array = -0.25 * normalized_chord_distribution_array
#     trailing_edge_array = 0.75 * normalized_chord_distribution_array
#     ax.plot(z_b_array, leading_edge_array, color="black")
#     ax.plot(z_b_array, trailing_edge_array, color="black")
#     ax.plot([z_b_array[0], z_b_array[0]], [leading_edge_array[0], trailing_edge_array[0]], color="black")
#     ax.plot([z_b_array[-1], z_b_array[-1]], [leading_edge_array[-1], trailing_edge_array[-1]], color="black")
#     for i in range(len(z_b_array)):
#         ax.plot([z_b_array[i], z_b_array[i]], [leading_edge_array[i], trailing_edge_array[i]], color="blue", linestyle=":")
#     ax.set_xlabel("$z/b$")
#     ax.set_xlim(-0.55, 0.55)
#     ax.set_ylabel("$x/b$")
#     ax.set_aspect("equal", adjustable="datalim")  # Fix aspect ratio
#     ax.invert_xaxis()
#     ax.invert_yaxis()
#     ax.set_title("Planform", fontsize=calculate_fontsize("Planform", ax.get_position().width * fig.get_size_inches()[0]))

#     # Add aileron positions (top-left overlay)
#     aileron_position = calc_aileron_position()  # Replace with your calculation
#     ax.plot(aileron_position[0][:, 0], aileron_position[0][:, 1], color="red", linestyle="--")
#     ax.plot(aileron_position[1][:, 0], aileron_position[1][:, 1], color="red", linestyle="--")

#     # Plot other distributions in remaining subplots
#     for i, (CL_distribution, dist_name, title) in enumerate(zip(CL_distributions, distribution_names, titles)):
#         row, col = divmod(i + 1, 2)  # Determine row and column index (skip first plot)
#         ax = axes[row, col]
#         ax.grid()
#         ax.plot(z_b_array, CL_distribution, label=dist_name)
#         ax.invert_xaxis()  # Match original formatting
#         ax.set_xlabel("$z/b$")
#         ax.set_ylabel(dist_name)
#         fontsize = calculate_fontsize(title, ax.get_position().width * fig.get_size_inches()[0])
#         ax.set_title(title, fontsize=fontsize)

#     # Add overall title and adjust layout
#     fig.suptitle("Lift Distributions", fontsize=18)
#     fig.tight_layout(rect=[0, 0, 1.0, 0.96])  # Avoid overlapping with suptitle
#     # plt.show()
#     plt.savefig("Lift_Distributions.pdf")

        # Create an animation
def update_frame(frame_idx):
    img = plt.imread(image_filenames[frame_idx])
    ax.imshow(img)
    ax.axis("off")  # Turn off axes for clean animation

def create_combined_distribution_figure(z_b_array, chord_distribution_array, CL_distributions, distribution_names, titles, filename):
    """
    Create a combined figure with multiple subplots, adjusted for better spacing and proper aspect ratio.
    Save the figure as an image file.
    """
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 12))
    plt.subplots_adjust(hspace=1.2, wspace=0.4)

    def calculate_fontsize(title, ax_width):
        max_fontsize = 16
        min_fontsize = 12
        title_length = len(title)
        fontsize = min(max_fontsize, max(min_fontsize, ax_width / title_length * 1.5))
        return fontsize

    # Planform plot
    ax = axes[0, 0]
    ax.grid()
    normalized_chord_distribution_array = normalize_chord_by_span(chord_distribution_array)
    leading_edge_array = -0.25 * normalized_chord_distribution_array
    trailing_edge_array = 0.75 * normalized_chord_distribution_array
    ax.plot(z_b_array, leading_edge_array, color="black")
    ax.plot(z_b_array, trailing_edge_array, color="black")
    ax.plot([z_b_array[0], z_b_array[0]], [leading_edge_array[0], trailing_edge_array[0]], color="black")
    ax.plot([z_b_array[-1], z_b_array[-1]], [leading_edge_array[-1], trailing_edge_array[-1]], color="black")
    for i in range(len(z_b_array)):
        ax.plot([z_b_array[i], z_b_array[i]], [leading_edge_array[i], trailing_edge_array[i]], color="blue", linestyle=":")
    ax.set_xlabel("$z/b$")
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylabel("$x/b$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title("Planform", fontsize=calculate_fontsize("Planform", ax.get_position().width * fig.get_size_inches()[0]))

    # Add other subplots
    for i, (CL_distribution, dist_name, title) in enumerate(zip(CL_distributions, distribution_names, titles)):
        row, col = divmod(i + 1, 2)
        ax = axes[row, col]
        ax.grid()
        ax.plot(z_b_array, CL_distribution, label=dist_name)
        ax.invert_xaxis()
        ax.set_xlabel("$z/b$")
        ax.set_ylabel(dist_name)
        fontsize = calculate_fontsize(title, ax.get_position().width * fig.get_size_inches()[0])
        ax.set_title(title, fontsize=fontsize)

    fig.suptitle("Lift Distributions", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1.0, 0.96])
    plt.savefig(filename)  # Save the figure as an image file
    plt.close(fig)  # Close the figure to free memory

def create_combined_time_plot(time_for_pbar, CL_array, CD_array, Cl_array, Cn_array, pbar_original):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()  # Flatten the axes for easier indexing

    # Data arrays and labels
    data = [CL_array, CD_array, Cl_array, Cn_array]
    labels = ["$C_L$ Total", "$C_{{D_i}}$ Total", "$C_l$ Total", "$C_n$ Total"]
    colors = ["blue", "orange", "green", "red"]

    for i, ax in enumerate(axs):
        # include grid lines
        ax.grid()
        # Plot the original data
        ax.plot(time_for_pbar, data[i], label=labels[i], color=colors[i])

        # Calculate and plot the mean as a second line
        mean_value = np.mean(data[i])
        ax.plot(time_for_pbar, [mean_value] * len(time_for_pbar), 
                linestyle="--", color="red", label=f"Mean: {mean_value:.2f}")

        # Set title to include the mean
        ax.set_title(f"Mean = {mean_value:.14f}")

        # Add axis labels and legend
        ax.set_xlabel("$\%$ of flap period")
        ax.set_ylabel(labels[i])
        # ax.legend()
    fig.suptitle("max pbar = " + str(abs(pbar_original)))
    # add a space below the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # Adjust layout for better spacing
    # fig.tight_layout()
    # create overall title
    plt.show()

if __name__ == "__main__":
    # initialize the cylinder object
    theta_array = calculate_thetas() # calculate the theta values
    z_b_array = calculate_zb(theta_array) # calculate the z/b values
    chord_distribution_array = calc_chord_dist(theta_array) # calculate the chord distribution array    
    washout_distribution_array = calc_washout_dist(theta_array) # calculate the washout distribution array
    Chi_distribution_array = calc_Chi_dist(theta_array) # calculate the aileron (Chi) distribution array
    # print("C_matrix")   
    C_matrix = calc_C_matrix(N_nodes, theta_array, chord_distribution_array) # Calculate the C matrix
    # for i in range(0, N_nodes):
        # print(C_matrix[i])
    # print("C_,matrix_end")
    # C_matrix_inv = np.linalg.inv(C_matrix) # Calculate the inverse of the C matrix
    
    right_side_for_fourier_coeff_a = np.ones((N_nodes)) # create the right side vector for the Fourier coefficient "a" -> all ones 
    a_fourier_coeff_vec = np.linalg.solve(C_matrix, right_side_for_fourier_coeff_a) # calculate the Fourier coefficient vector a using the right side vector
    b_fourier_coeff_vec = np.linalg.solve(C_matrix, washout_distribution_array) # calculate the Fourier coefficient vector b using the washout distribution, omega 
    # print("b_fourier_coeff_vec")
    # for i in range(0, N_nodes):
        # print(b_fourier_coeff_vec[i])
    c_fourier_coeff_vec = np.linalg.solve(C_matrix, Chi_distribution_array) # calculate the Fourier coefficient vector c using the aileron distribution
    right_side_for_fourier_coeff_d = abs(np.cos(theta_array)) # calculate the Fourier coefficient vector d using the cosine of theta
    d_fourier_coeff_vec = np.linalg.solve(C_matrix, right_side_for_fourier_coeff_d) # calculate the Fourier coefficient vector d using the right side vector
    
    # make a loop based on pbar values
    pbar_original = pbar
    CL_array = []
    CD_array = []
    Cl_array = []
    Cn_array = []
    image_filenames = []  # List to store image filenames
    time_for_pbar = np.linspace(0, 2*np.pi, 50)
    # for i in range(len(time_for_pbar)):
    for i in tqdm(range(len(time_for_pbar)), desc="Calculating Forces and Moments"):
        pbar = pbar_original*np.sin(time_for_pbar[i])
        # print("pbar:", pbar)
        A_alpha = a_fourier(a_fourier_coeff_vec) # calculate the Fourier coefficient due to angle of attack
        A_washout = b_fourier(b_fourier_coeff_vec) # calculate the Fourier coefficient due to washout distribution
        A_aileron = c_fourier(c_fourier_coeff_vec) # calculate the Fourier coefficient due to aileron distribution
        A_roll_rate = d_fourier(d_fourier_coeff_vec) # calculate the Fourier coefficient due to roll rate
        A_total, A_symmetric, A_asymmetric = A_fourier_total(A_alpha, A_washout, A_aileron, A_roll_rate) # calculate the total Fourier coefficient
        CL_total = calc_CL(A_total) # calculate the total lift coefficient
        CL_array.append(CL_total)
        CL_total_distribution = calc_CL_distribution(A_total, theta_array) # calculate the total lift distribution
        CDi_total = calc_CDi(A_total) # calculate the total induced drag coefficient
        CD_array.append(CDi_total)
        Cl_total = calc_Cl(A_total) # calculate the total rolling moment
        Cl_array.append(Cl_total)
        Cn_total = calc_Cn(A_total) # calculate the total normal force
        Cn_array.append(Cn_total) 

        # print("Total")
        # print("CL:", CL_total)
        # print("CDi:", CDi_total)
        # print("Cl:", Cl_total)
        # print("Cn:", Cn_total)

        CL_from_alpha = calc_CL(A_alpha) # calculate the lift coefficient due to angle of attack
        CL_alpha_distribution = calc_CL_distribution(A_alpha, theta_array) # calculate the lift distribution due to angle of attack
        CDi_from_alpha = calc_CDi(A_alpha) # calculate the induced drag coefficient due to angle of attack
        Cl_from_alpha = calc_Cl(A_alpha) # calculate the rolling moment due to angle of attack
        Cn_from_alpha = calc_Cn(A_alpha) # calculate the normal force due to angle of attack
        # print("Alpha component")
        # print("CL:", CL_from_alpha)
        # print("CDi:", CDi_from_alpha)
        # print("Cl:", Cl_from_alpha)
        # print("Cn:", Cn_from_alpha)
        
        CL_from_washout = calc_CL(A_washout) # calculate the lift coefficient due to washout distribution
        CL_washout_distribution = calc_CL_distribution(A_washout, theta_array) # calculate the lift distribution due to washout distribution
        CDi_from_washout = calc_CDi(A_washout) # calculate the induced drag coefficient due to washout distribution
        Cl_from_washout = calc_Cl(A_washout) # calculate the rolling moment due to washout distribution
        Cn_from_washout = calc_Cn(A_washout) # calculate the normal force due to washout distribution
        # print("Washout component")
        # print("CL:", CL_from_washout)
        # print("CDi:", CDi_from_washout)
        # print("Cl:", Cl_from_washout)
        # print("Cn:", Cn_from_washout)

        CL_from_aileron = calc_CL(A_aileron) # calculate the lift coefficient due to aileron distribution
        CL_aileron_distribution = calc_CL_distribution(A_aileron, theta_array) # calculate the lift distribution due to aileron distribution
        CDi_from_aileron = calc_CDi(A_aileron) # calculate the induced drag coefficient due to aileron distribution
        Cl_from_aileron = calc_Cl(A_aileron) # calculate the rolling moment due to aileron distribution
        Cn_from_aileron = calc_Cn(A_aileron) # calculate the normal force due to aileron distribution
        # print("Aileron component")
        # print("CL:", CL_from_aileron)
        # print("CDi:", CDi_from_aileron)
        # print("Cl:", Cl_from_aileron)
        # print("Cn:", Cn_from_aileron)

        CL_from_roll_rate = calc_CL(A_roll_rate) # calculate the lift coefficient due to roll rate
        CL_roll_rate_distribution = calc_CL_distribution(A_roll_rate, theta_array) # calculate the lift distribution due to roll rate
        CDi_from_roll_rate = calc_CDi(A_roll_rate) # calculate the induced drag coefficient due to roll rate
        Cl_from_roll_rate = calc_Cl(A_roll_rate) # calculate the rolling moment due to roll rate
        Cn_from_roll_rate = calc_Cn(A_roll_rate) # calculate the normal force due to roll rate
        # print("Roll rate component")
        # print("CL:", CL_from_roll_rate)
        # print("CDi:", CDi_from_roll_rate)
        # print("Cl:", Cl_from_roll_rate)
        # print("Cn:", Cn_from_roll_rate)

        # Calculate the total symmetric and asymmetric lift coefficients
        symmetric_CL = calc_CL(A_symmetric) # calculate the symmetric lift coefficient
        CL_symmetric_distribution = calc_CL_distribution(A_symmetric, theta_array) # calculate the symmetric lift distribution
        symmetric_CDi = calc_CDi(A_symmetric) # calculate the symmetric induced drag coefficient
        symmetric_Cl = calc_Cl(A_symmetric) # calculate the symmetric rolling moment
        symmetric_Cn = calc_Cn(A_symmetric) # calculate the symmetric normal force
        # print("Symmetric")
        # print("CL:", symmetric_CL)
        # print("CDi:", symmetric_CDi)
        # print("Cl:", symmetric_Cl)
        # print("Cn:", symmetric_Cn)

        asymmetric_CL = calc_CL(A_asymmetric) # calculate the asymmetric lift coefficient
        CL_asymmetric_distribution = calc_CL_distribution(A_asymmetric, theta_array) # calculate the asymmetric lift distribution
        asymmetric_CDi = calc_CDi(A_asymmetric) # calculate the asymmetric induced drag coefficient
        asymmetric_Cl = calc_Cl(A_asymmetric) # calculate the asymmetric rolling moment
        asymmetric_Cn = calc_Cn(A_asymmetric) # calculate the asymmetric normal force
        # print("Asymmetric")
        # print("CL:", asymmetric_CL)
        # print("CDi:", asymmetric_CDi)
        # print("Cl:", asymmetric_Cl)
        # print("Cn:", asymmetric_Cn)

        # Generate filenames for each frame
        filename = f"frame_{i:03d}.png"
        image_filenames.append(filename)

        create_combined_distribution_figure(
            z_b_array,
            chord_distribution_array,
            [
                CL_total_distribution,
                CL_alpha_distribution,
                CL_washout_distribution,
                CL_aileron_distribution,
                CL_roll_rate_distribution,
                CL_symmetric_distribution,
                CL_asymmetric_distribution,
            ],
            [
                "$C_L$ Total",
                "$C_L$ Alpha Component",
                "$C_L$ Washout Component",
                "$C_L$ Aileron Component",
                "$C_L$ pbar Component",
                "$C_L$ Symmetric Component",
                "$C_L$ Asymmetric Component",
            ],
            [
                f"$C_L$={CL_total:.14f} $C_{{D_i}}$={CDi_total:.14f} $C_l$={Cl_total:.14f} $C_n$={Cn_total:.14f}",
                f"$C_L$={CL_from_alpha:.14f} $C_{{D_i}}$={CDi_from_alpha:.14f} $C_l$={Cl_from_alpha:.14f} $C_n$={Cn_from_alpha:.14f}",
                f"$C_L$={CL_from_washout:.14f} $C_{{D_i}}$={CDi_from_washout:.14f} $C_l$={Cl_from_washout:.14f} $C_n$={Cn_from_washout:.14f}",
                f"$C_L$={CL_from_aileron:.14f} $C_{{D_i}}$={CDi_from_aileron:.14f} $C_l$={Cl_from_aileron:.14f} $C_n$={Cn_from_aileron:.14f}",
                f"$C_L$={CL_from_roll_rate:.14f} $C_{{D_i}}$={CDi_from_roll_rate:.14f} $C_l$={Cl_from_roll_rate:.14f} $C_n$={Cn_from_roll_rate:.14f}",
                f"$C_L$={symmetric_CL:.14f} $C_{{D_i}}$={symmetric_CDi:.14f} $C_l$={symmetric_Cl:.14f} $C_n$={symmetric_Cn:.14f}",
                f"$C_L$={asymmetric_CL:.14f} $C_{{D_i}}$={asymmetric_CDi:.14f} $C_l$={asymmetric_Cl:.14f} $C_n$={asymmetric_Cn:.14f}",
            ],
            filename
        )

    # Create an animation
    fig, ax = plt.subplots(figsize=(10, 8))
    ani = animation.FuncAnimation(fig, update_frame, frames=len(image_filenames), interval=100) # fig is the figure, update_frame is the function to update the frame, frames is the number of frames, interval is the time between frames in milliseconds

    # Save the animation as a video or GIF
    ani.save("Lift_Distributions_Animation.gif", writer="pillow", fps=10)

    # Cleanup saved frames
    for filename in image_filenames:
        os.remove(filename)
    
    CL_array = np.array(CL_array)
    CD_array = np.array(CD_array)
    Cl_array = np.array(Cl_array)
    Cn_array = np.array(Cn_array)
    normalized_time_for_pbar = time_for_pbar/(2*np.pi)

    create_combined_time_plot(normalized_time_for_pbar, CL_array, CD_array, Cl_array, Cn_array, pbar_original)
    