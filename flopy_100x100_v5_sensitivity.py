# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:46:10 2024

@author: Jorrit

This script is specifically designed for the sensitivity analysis of the pumping rate
It can also be used for sensitivity analysis of other parameters, with some simple modifications
"""



#%%
# Import necessary modules
import numpy as np
import flopy
import os
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import tools as tool
import time
import copy
#%%
start_time = time.time()

# Folders
modelname = "100x100_mf"
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

# Sensitivty analysis
plot_flux = True                   # Toggle to make plots of the flux between the 8th and 9th layer
pump_mltyply = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 10] # List with values for the sensitivity analysis. Values are multiplied with the pumping rates
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

# Get the horizontal and vertical hydraulic conductivity
hk = matrix_dict['hk']
vk = matrix_dict['vk']
# Add the layer properties 
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vk, ipakcb=53)

# Recharge and well
rch = flopy.modflow.ModflowRch(mf, nrchop=1, rech=rech) 

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

#%%
# Make empty array for the nett bottom flux
flux_array = np.zeros(len(pump_mltyply))
# Make empty array for the positive (downward) bottom flux
posflux_array = np.zeros(len(pump_mltyply))
# Iterate over each parameter value in the array for the sensitivity analysis
for i, factor in enumerate(pump_mltyply):
    # Make a deep copy of the standard pumping rate, because otherwise it will be altered in place
    m_lrcQ = copy.deepcopy(lrcQ)
    # Multiply the pumping rate for each well in lrcQ by the pumping rate multiplier
    for sublist in m_lrcQ[1]:
        sublist[-1] *= factor
    
    # Include the pumping rate in the model
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=m_lrcQ)

    # Write the input and run the MODFLOW model
    mf.write_input()
    success, buff = mf.run_model()
    if not success:
        raise Exception("MODFLOW did not terminate normally.")
     
    # Write the input and run the MT3DMS model
    mt.write_input()
    suc6, buff2 = mt.run_model()
    
    # Get the object file containing the hydraulic head distribution
    hds = bf.HeadFile(f"{modelname}.hds")
    
    # Get the object file containing the concentration
    ucnobj = bf.UcnFile('MT3D001.UCN')

    # Read the head and concentration files
    cbc = flopy.utils.CellBudgetFile(f"{modelname}.cbc")
    # Get the stress period and time step times
    times = ucnobj.get_times()

    # Get the cell budget data from the flow of the lower cell face of the last time step
    flow_lower_face = cbc.get_data(text='FLOW LOWER FACE', totim=times[-1])[0]
    # Get the concentrations of the last time step
    conc_data = ucnobj.get_data(totim=times[-1])
    
    # Initialize flux array
    flux = np.zeros(flow_lower_face.shape)
    
    # Loop through layers and calculate flux
    for k in range(flow_lower_face.shape[0] - 1):
        # COntaminant flux is calculated as water flux from upper to lower layer multiplied with the average concentration between the layers
        flux[k, :, :] = flow_lower_face[k, :, :] * (conc_data[k, :, :] + conc_data[k+1, :, :]) / 2
    
    # Get the positive flux only; only the values where water flows from upper to lower layer
    pos_flux = np.sum(np.where(flux[8,:,:] > 0, flux[8,:,:], 0))    
    # The total (nett) flux between the 8th and 9th layer is the sum of all fluxes
    total_flux = np.sum(flux[8,:,:])
    # Save the positive and total flux in their respective arrays
    flux_array[i] = total_flux
    posflux_array[i] = pos_flux

    # Optionally, plot the flux for visualization
    if plot_flux:
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(flux[7, :, :], cmap='viridis', extent=(0, flux.shape[2], 0, flux.shape[1]))
        plt.colorbar(label='Contaminant Flux')
        plt.title(f'Contaminant Flux at Time {times[-1]}, for pump factor {factor}')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()

print(total_flux)

#%% Plotting

fontsize = 9
print(flux_array)
plt.figure(dpi=300)
# Plot the array with pumping multipliers against the nett flux
plt.plot(pump_mltyply, flux_array, '-o', color = 'blue')
plt.xlabel("Pumping rate multiplication factor", fontsize = fontsize)
plt.ylabel("Total nett flux from $8^{th}$ to $9^{th}$ layer [$kg\,m^{-3} d^{-1}$]", fontsize = fontsize)
plt.show()

fontsize = 9
print(flux_array)
plt.figure(dpi=300)
# Plot the array with pumping multipliers against the positive flux
plt.plot(pump_mltyply, posflux_array, '-o', color = 'blue')
plt.xlabel("Pumping rate multiplication factor", fontsize = fontsize)
plt.ylabel("Total flux from $8^{th}$ to $9^{th}$ layer [$kg\,m^{-3} d^{-1}$]", fontsize = fontsize)
plt.show()