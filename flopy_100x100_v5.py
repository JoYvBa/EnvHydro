"""
Created on Sat Jun 11 12:34:56 2024

@author: Jorrit

This script contains the basic FloPy code
"""

#%%

# Import necessary modules
import numpy as np
import flopy
import os
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import gstools as gs
import tools as tool
import time
from matplotlib.ticker import FormatStrFormatter

#%% Parameters

start_time = time.time()

# Folders
modelname = "100x100_mf" # Name of the model
folder =  r"./ASCII_matricces/100x100_Naph" # Folder with the PMWIN ASCII tables of the right scenario

# Fill in your own path to the MODFLOW executable here!!!
mf = flopy.modflow.Modflow(modelname, exe_name=r"C:\Program Files (x86)\PM8\modflow2005\mf2005")
mt = flopy.mt3d.Mt3dms(modelname, modflowmodel=mf, exe_name=r"C:\Program Files (x86)\PM8\mt3dms\mt3dms5b.exe")

# File parameters
# In names, put the abreviations for the input types of the model, in alphabetic order!
names = ["botm", "ghb", "hfb", "hk", "ibound", "prsity", "sconc", "spd1", "spd2", "strt", "vk"]
# Put the names of the elements containing stress period data for the solutes
s_period_names = ["spd1", "spd2"]

# Geometry parameters
Lx = 1000.0         # Extent of the model in the x direction [m]
Ly = 1000.0         # Extent of the model in the y direction [m]
ztop = 145          # Height of the top layer [m]
zbot = 0            # Height of the bottom layer [m]
nlay = 10           # Number of model layers 
nrow = 100          # Number of model rows
ncol = 100          # Number of model columns
delr = Lx / ncol    # Cell size in the x direction [m]
delc = Ly / nrow    # Cell size in the y direction [m]

# Hydraulic parameters
## Horizontal flow barrier and general head boundary
k_hfb = 0.001 # Hydraulic conductivity of the horizontal flow barrier [m/day]
k_ghb = 10000000 # Hydraulic conductance for general head boundary [m2/day]

## Well and recharge
rech = 0.00097 # Recharge to the top layer [m/day]
lrcQ = {0: [[2, 35, 55, 0], [2, 38, 48, 0], [2, 51, 57, 0]], 1: [[2, 35, 55, -84], [2, 38, 48, -96], [2, 51, 57, -240]]} # Well pumping rate as; {stress period: [[layer, row, col, flux]]}, with flux in [m3/d]

# Transport parameters
transport_toggle = False           # Toggle for running with or without contaminant transport
perlen = np.array([40150, 12410])  # Length of each stress period
nstp = np.array([8, 8])            # Time steps in each stress period
icbund = 1                         # Concentration boundary conditions, 1 means active cells, 0 inactive cells and -1 constant concentration 
itype = "CC"                       # Source/Sink type, CC standing for constant concentration
percel = 0.75                      # Courant number
mixelm = 0                         # Advection solution method, 0 stands for standard finite-difference method
al = 5                             # Longitudinal dispersivity
trpt = 0.1                         # Ratio between horizontal transverse dispersivity and longitudinal dispersivity
trpv = 0.01                        # Ratio between vertical transverse dispersivity and longitudinal dispersivity
dmcoef = 0.000121                  # Diffusion coefficient
Kd = 0.0099                        # Distribution coefficent of the adsorping solute

# Stochastic parameters
stoch_toggle = True                # Turn on or off to allow stochastic field generation
stoch_lay = [4, 5, 6, 7]           # The layer numbers (starting at 0) of which the hydraulic conductivity has to be a stochastic variable
var = [0.10, 0.10, 0.10, 0.10]     # Variance of the stochastic variable per layer
len_scale = 75                     # Length scale of the stochastic variable
n_real = 1                         # Number of stochastic realizations

# Visual preferences
print_h = True                     # Toggle to make plots of the head distribution
vector = True                      # Toggle to include flow vectors in the head distribution plots
contour_n = 60                     # Number of contours in 
layer_h = [0,2,4,8]                # Layer indecces to plot the head distribution of 
layer_conc = [0,2,4,8]             # Layer indecces to plot the concentration distribution of
fontsize = 14                      # Title font size of the plots
fontsize_s = 12                    # Tick font size of the plots

#%% Running the model

# Join folder name with file name for all ASCII matrices in the folder
filepaths = [os.path.join(folder, name) for name in os.listdir(folder)]

# Make an empty dictionary to put the input files into
matrix_dict = {}

# Loop over the list with ASCII matrices to convert them into numpy arrays and combine the layers for each parameter value
for i, location in enumerate(filepaths):
    # If the iteration acount is fully divisible by the amount of layers (meaning the files belong to a new parameter), make an empty 3D array for the parameter.
    if (i % nlay) == 0:
        n = 0
        mat = np.zeros((nlay, nrow, ncol))
        # NOTE: As this code is based on the fact that files are sorted on alphabetical order, files for layer 10 come before layer 1 (using this file format).
        # thus, the first file of a new parameter is saved into the last positions of the matrix. This should be changed when using a model with a different
        # amount of layers than 10. Could be solved by naming files with _l01_, since 01, 02, 03 etc. will alphabetically be before 10, 11, 12 etc.
        mat[nlay-1, :, :] = tool.read(file=location, x=ncol, y=nrow)
    else:
        # If it is not the first file of a parameter, put it in the position of the current layer
        mat[n - 1, :, :] = tool.read(file=location, x=ncol, y=nrow)
    
    # If the next file would be fully divisible by the amount of layers, save the matrix in the dictionary with the right key.
    if ((i + 1) % nlay) == 0:
        matrix_dict[names[i // nlay]] = mat
    
    n += 1

# Set the bottom of layers botm to the values from the dictionary
botm = matrix_dict["botm"]

# Add the decretization to the model
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm, perlen=perlen, nstp = nstp, nper = len(perlen))

# Get the boundary values and initial concentrations from the dictionary
ibound = matrix_dict['ibound']
strt = matrix_dict['strt']

# Add the basic flow package
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

# Add the recharge and wells
rch = flopy.modflow.ModflowRch(mf, nrchop=1, rech=rech) 
wel = flopy.modflow.ModflowWel(mf, stress_period_data=lrcQ)

# Get the horizontal flow barrier from the dictionary
hfb = matrix_dict["hfb"]
# Convert the hfb data from a matrix to a list
hfb_out = tool.hfbconvert(hfb, k_hfb)

# Add the horizontal flow barrier 
hfb_mod = flopy.modflow.ModflowHfb(mf, hfb_data=hfb_out)

# Get the general head boundary from the dictionary
ghb = matrix_dict["ghb"]
# Convert the ghb data from a matrix to a list
ghb_out = tool.ghbconvert(ghb, k_ghb)
# Add the general head boundary
ghb_mod = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_out)

# Output control
stress_period_data = {}
# For every stress period and time step, save the information of choice
for kper in range(len(perlen)):
    for kstp in range (nstp[kper]):
        stress_period_data[(kper, kstp)] = [
            "save head",
            "save drawdown",
            "save budget",
            "print head",
            "print budget",
        ]

# Add output control to the model
oc = flopy.modflow.ModflowOc(mf, stress_period_data= stress_period_data, compact=True)

# Add the Preconditioned Conjugate Gradient Package; controling inner iterations, convergence, etc.
pcg = flopy.modflow.ModflowPcg(mf, hclose=1e-4, rclose=1e-4, relax=0.97)


# If transporting is moddeld;
if transport_toggle:
    
    # Give the name of the link file between MODFLOW and MT3DMS
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name="mt3d_link.ftl")
    
    # Make an empty dictionary
    stress_period_data_mt = {}
    # Fill the dictionary with a list of source concentrations for each stress period
    for i, spd_name in enumerate(s_period_names):
        stress_period_data_mt[i] = tool.s_period_con_convert(spd=matrix_dict[spd_name], itype=itype)
    
    # Get the porosity from the dictionary
    prsity = matrix_dict["prsity"]
    # Get the start concentration from the dictionary
    sconc = matrix_dict["sconc"]
    
    # Add the basic transport package
    btn = flopy.mt3d.Mt3dBtn(mt, prsity=prsity, icbund=1, sconc=sconc, perlen=perlen, nstp=nstp)
    # Add the advection package
    adv = flopy.mt3d.Mt3dAdv(mt, percel=percel, mixelm=mixelm)
    # Add the dispersion package
    dsp = flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt, trpv=trpv, dmcoef=dmcoef)
    # Add the source/sink and mixing pakcage
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=stress_period_data_mt)
    # Add the reaction package
    rct = flopy.mt3d.Mt3dRct(mt, isothm = 1, sp1 = Kd, srconc = 0, igetsc = 0)
    # Add the Generalized Conjugate Gradient Package Class of MT3DMS
    gcg = flopy.mt3d.Mt3dGcg(mt, cclose=1e-6)

# Either enter a loop for each stochastic iteration or just run the model once if no stochastic modelling is enabled
if stoch_toggle:
    # Make empty lists for the concentration and hydraulic head
    head_list = []
    conc_list = []
    
    # Loop over the amount of realisations
    for n in range(n_real):
        # Copy the horizontal and vertical hydraulic conductivity from the dictionary (needs to be copyed since otherwise it would make changes in place)
        hk = matrix_dict['hk'].copy()
        vk = matrix_dict['vk'].copy()
        # Make x and y arrays over the length of the model with the appropiated step size
        x = np.linspace(0, Lx, nrow)
        y = np.linspace(0, Ly, ncol)
    
        # Iterate over each layer which should get a Gaussian random field for the hydraulic conductivity
        for i, layer in enumerate(stoch_lay):
            # Get the Gaussian covariance model with the desired variance and length scale
            model = gs.Gaussian(dim=2, var=var[i], len_scale=len_scale)
            # Generate random values with model on a set seed, different for every layer and iteration
            srf = gs.SRF(model, seed=10170520 + n + i * 200)
            # Shape the Gaussian random field to the dimension of hte model
            field = srf.structured([x, y])
            # Use the exisiting hydraulic conductivities as mean for the rnadom field
            hk[layer, :, :] = hk[layer, :, :] * np.exp(field)
            vk[layer, :, :] = vk[layer, :, :] * np.exp(field)
        
        # Set the layer property flow package
        lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vk, ipakcb=53)
        
        # Write all input for the MODFLOW model
        mf.write_input()
        # Run the MODFLOW model
        success, buff = mf.run_model()
        # If the run_model() gives a False for the succes parameter, something went wrong and an error is given
        if not success:
            raise Exception("MODFLOW did not terminate normally.")
            
        # Save the object with hydraulic heads
        hds = bf.HeadFile(f"{modelname}.hds")
        
        
        if n == 0:
            # Get the model times only at the first realisation, since it is the same for the rest
            times = hds.get_times()
        # Get the head distribution for the last time period of the last stress period
        head = hds.get_data(totim=times[-1])
        # Add that hydraulic head to the list with heads
        head_list.append(head)
        
        # If transport is modeled
        if transport_toggle:
            # Write the MT3DMS input
            mt.write_input()
            # Run the MT3DMS model
            suc6, buff2 = mt.run_model()
            # Since for unknown reasons, the succes status of MT3DMS would always return False, even though all data was properly generated, no check is done for the succes status like was done for MODFLOW. If anyone knows why False is always returned, please let me know!
            
            # Get the object containing concentration information
            ucnobj = bf.UcnFile('MT3D001.UCN')
            # Get the concentration for the last stress period and time step
            conc = ucnobj.get_data(totim=times[-1])
            # Append the concentration to the list
            conc_list.append(conc)
    
    # Take the average head distribution over all iterations
    head_stack = np.stack(head_list)
    average_head = np.mean(head_stack, axis=0)
    
    # Take the average concentration over all iterations
    if transport_toggle:
        conc_stack = np.stack(conc_list)
        average_conc = np.mean(conc_stack, axis=0)
    
    # Set the head and conc variables to the average values from the model run
    head = average_head
    if transport_toggle:
        conc = average_conc
    
else:
    # Take the horizontal and vertical hydraulic conductivity from the dictionary
    hk = matrix_dict['hk']
    vk = matrix_dict['vk']
    
    # Set the layer property flow package
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vk, ipakcb=53)
    
    # Write the input and run the MODFLOW model
    mf.write_input()
    success, buff = mf.run_model()
    if not success:
        raise Exception("MODFLOW did not terminate normally.")
    
    # Write the input and run the MT3DMS model
    if transport_toggle:
        mt.write_input()
        suc6, buff2 = mt.run_model()
    
    # Save the head and concentration distribution objects
    hds = bf.HeadFile(f"{modelname}.hds")
    times = hds.get_times()
    
    if transport_toggle:
        ucnobj = bf.UcnFile('MT3D001.UCN')
        times = ucnobj.get_times()


    
#%% Plotting the model


# If you want to show the hydraulic heads
if print_h:
# For each stress period    
    for i, t in enumerate(times[0:-1:nstp[0]]):
        # For each layer you want to plat
        for layer in layer_h:
            # Get the hydraulic head for that time
            head = hds.get_data(totim=t)
            # If any hydraulic head is below -900, it is is -999.99, which occurs at no flow boundary conditions. It is set to a NaN value instead to not intervere with plotting
            head[head < -900] = np.nan
            
            # Get the cell budget file
            cbb = bf.CellBudgetFile(f"{modelname}.cbc")
            # And get the flow on the right and front face, used for flow direction
            frf = cbb.get_data(text='FLOW RIGHT FACE', totim=t)[0]
            fff = cbb.get_data(text='FLOW FRONT FACE', totim=t)[0]
            
            # Initialize the figure
            fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
            # Set it as a mapview
            mapview = flopy.plot.PlotMapView(model=mf, layer=layer)
            
            # If you want to plot the vectors
            if vector:
                # Add vector field with the inbuild FloPy function
                qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge((frf, fff, np.zeros(frf.shape)), mf)
                # Plot every second vector in the x and y directions (istep and jstep)
                quiver = mapview.plot_vector(qx, qy, istep=2, jstep=2, normalize=True, color='red', scale=50, headwidth=5, headlength=5, headaxislength=6, pivot='middle')
            
            # Make a contour map for the head distribution
            contour_set = mapview.contour_array(head, levels=np.linspace(np.nanmin(head), np.nanmax(head), contour_n))
            # Add a colour bar
            cbar = plt.colorbar(contour_set, ax=ax, shrink=0.5, aspect=5)
            # Change the colourbar label and axis ticks
            cbar.set_label("head [m]", fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize_s) 
            # Round colour tick values to two decimals
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
            # Change axis tick labels size
            ax.tick_params(axis='both', which='major', labelsize=fontsize_s)
            
            # Give the plot a title, x and y axis label
            plt.title(f'Stress period {i+1} - Layer {layer+1}', fontsize=fontsize)
            plt.xlabel("Model x extent [m]", fontsize=fontsize)
            plt.ylabel("Model y extent [m]", fontsize=fontsize)
            plt.show()
#%%
if transport_toggle:
    # Plot concentration at the last time step !!!!(this will need adjustment to your own preferences and input!!!!)
    for i, t in enumerate(times[nstp[0]-1 :sum(nstp) + 1:nstp[-1]]):
        # Get concentration data
        conc = ucnobj.get_data(totim=t)
        for n in layer_conc:
            # Initiate the plot
            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            # Make a 2D matrix visualization with the proper dimensions
            img = ax.imshow(conc[n, :, :], cmap='magma', extent=[0, Ly, 0, Lx])  # Set extent from 0 to 1000 in both x and y directions
            # Include a colourbar
            cbar = plt.colorbar(img, ax=ax, shrink=0.5, aspect=5)
            # Change the colourbar label and axis ticks
            cbar.set_label("Concentration [$kg\,m^{-3}$]", fontsize = fontsize)
            cbar.ax.tick_params(labelsize=fontsize_s)  # Change colorbar tick labels size
    
            # Change axis tick labels size
            ax.tick_params(axis='both', which='major', labelsize=fontsize_s)
            
            # Give the plot a title, x and y axis label
            plt.title(f'Stress period {i+1} - Layer {n+1}', fontsize=fontsize)
            plt.xlabel("model x extent [m]", fontsize=fontsize)
            plt.ylabel("Model y extent [m]", fontsize=fontsize)
            plt.show()
#%%
# Show the run time, so next run, you know for how long you can take a coffee break
print("It took %s seconds to run the model" % np.round(time.time() - start_time, 1))
