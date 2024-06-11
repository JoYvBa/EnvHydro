# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:46:10 2024

@author: Jorrit
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
import copy
#%%
start_time = time.time()

# Folders
modelname = "100x100_mf"
# Fill in your own path to the MODFLOW executable here!!!
mf = flopy.modflow.Modflow(modelname, exe_name=r"C:\Program Files (x86)\PM8\modflow2005\mf2005")
mt = flopy.mt3d.Mt3dms(modelname, modflowmodel=mf, exe_name=r"C:\Program Files (x86)\PM8\mt3dms\mt3dms5b.exe")

# File parameters
names = ["botm", "ghb", "hfb", "hk", "ibound", "prsity", "sconc", "spd1", "spd2", "strt", "vk"]
s_period_names = ["spd1", "spd2"]

# Scenario parameters
ghb_toggle = True # Enables a general head boundary over constant head boundary
hfb_toggle = True # Enables a horizontal flow barrier over a no flow barrier

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
perlen = np.array([40150, 12410])  # Length of each stress period
nstp = np.array([8, 8])            # Time steps in each stress period
icbund = 1                         # Concentration boundary conditions, 1 means active cells, 0 inactive cells and -1 constant concentration 
itype = "CC"                       # Source/Sink type, CC standing for constant concentration
percel = 0.75                      # Courant number
mixelm = 0                         # Advection solution method, 0 stands for standard finite-difference method
al = 5                             # Longitudinal dispersivity
trpt = 0.1                         # Ratio between horizontal transverse dispersivity and longitudinal dispersivity
trpv = 0.01                        # Ratio between vertical transverse dispersivity and longitudinal dispersivity
dmcoef = 0.0000657                 # Diffusion coefficient
Kd = 5e-3                         # Distribution coefficent of the adsorping solute

# Stochastic parameters
stoch_toggle = False             # Turn on or off to allow stochastic field generation
stoch_lay = [4, 5, 6, 7]          # The layer numbers (starting at 0) of which the hydraulic conductivity has to be a stochastic variable
var = [0.10, 0.10, 0.10, 0.10]   # Variance of the stochastic variable per layer
len_scale = 75                   # Length scale of the stochastic variable
n_real = 1                       # Number of stochastic realizations

# Visual preferences
vector = False
contour_n = 50
layer_h = [0,2,4,8]              # Layer indecces to plot the head distribution of 
layer_conc = [0,2,4,8]

# Sensitivty analysis
pump_mltyply = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 10]
#%%

# Determine the scenario
folder =  r"./ASCII_matricces/100x100_MT"

'''
if ghb_toggle and hfb_toggle:
    folder = r"./ASCII_matricces/100x100_10l_ghb_hfb"
    print("Showing scenario with general head boundary and horizontal flow barrier")
elif hfb_toggle and not ghb_toggle:
    folder = r"./ASCII_matricces/100x100_10l_chead_hfb"
    print("Showing scenario with constant head boundary and horizontal flow barrier")
elif not hfb_toggle and not ghb_toggle:
    folder = r"./ASCII_matricces/100x100_10l_noflo_chead"
    print("Showing scenario with constant head boundary and no flow boundary")
elif ghb_toggle and not hfb_toggle:
    folder = r"./ASCII_matricces/100x100_10l_noflo_chead"
    print("Showing scenario with general head boundary and no flow boundary")
'''

# Join folder name with file name for all ASCII matrices in the folder
filepaths = [os.path.join(folder, name) for name in os.listdir(folder)]

matrix_dict = {}

# Loop over the list with ASCII matrices to convert them into numpy arrays and combine the layers for each parameter value
for i, location in enumerate(filepaths):
    if (i % nlay) == 0:
        n = 0
        mat = np.zeros((nlay, nrow, ncol))
        mat[nlay-1, :, :] = tool.read(file=location, x=ncol, y=nrow)
    else:
        mat[n - 1, :, :] = tool.read(file=location, x=ncol, y=nrow)
    
    if ((i + 1) % nlay) == 0:
        matrix_dict[names[i // nlay]] = mat
    
    n += 1

botm = matrix_dict["botm"]

# Discretization
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm, perlen=perlen, nstp = nstp, nper = len(perlen))

lmt = flopy.modflow.ModflowLmt(mf, output_file_name="mt3d_link.ftl")

if ghb_toggle and not hfb_toggle:
    matrix_dict["ibound"][matrix_dict["ibound"] == -1] = 1

# Basic package
ibound = matrix_dict['ibound']
strt = matrix_dict['strt']
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

# Recharge and well
rch = flopy.modflow.ModflowRch(mf, nrchop=1, rech=rech) 


# General head boundary and horizontal flow barrier 
if hfb_toggle:
    hfb = matrix_dict["hfb"]
    hfb_out = tool.hfbconvert(hfb, k_hfb)
    
    # Validate HFB data
    if not tool.validate_hfb_data(hfb_out, nlay, nrow, ncol):
        raise ValueError("Invalid HFB data detected.")
    
    hfb_mod = flopy.modflow.ModflowHfb(mf, hfb_data=hfb_out)

if ghb_toggle:
    ghb = matrix_dict["ghb"]
    ghb_out = tool.ghbconvert(ghb, k_ghb)
    ghb_mod = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_out)

# Output control
stress_period_data = {}
for kper in range(len(perlen)):
    for kstp in range (nstp[kper]):
        stress_period_data[(kper, kstp)] = [
            "save head",
            "save drawdown",
            "save budget",
            "print head",
            "print budget",
        ]
oc = flopy.modflow.ModflowOc(mf, stress_period_data= stress_period_data, compact=True)
# Preconditioned Conjugate Gradient Package
pcg = flopy.modflow.ModflowPcg(mf, hclose=1e-4, rclose=1e-4, relax=0.97)

# Gaussian distributed horizontal hydraulic conductivity

# MT3DMS

## Data pre-processing

stress_period_data_mt = {}
for i, spd_name in enumerate(s_period_names):
    stress_period_data_mt[i] = tool.s_period_con_convert(spd=matrix_dict[spd_name], itype=itype)
stress_period_data_mt[1] = [[0, 5, 5, 0, -1]]

prsity = matrix_dict["prsity"]
sconc = matrix_dict["sconc"]

## Functions 
btn = flopy.mt3d.Mt3dBtn(mt, prsity=prsity, icbund=1, sconc=sconc, perlen=perlen, nstp=nstp)
adv = flopy.mt3d.Mt3dAdv(mt, percel=percel, mixelm=mixelm)
dsp = flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt, trpv=trpv, dmcoef=dmcoef)
ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=stress_period_data_mt)
rct = flopy.mt3d.Mt3dRct(mt, isothm = 1, sp1 = Kd, srconc = 0, igetsc = 0)
gcg = flopy.mt3d.Mt3dGcg(mt, cclose=1e-6)

#%%
flux_array = np.zeros(len(pump_mltyply))
posflux_array = np.zeros(len(pump_mltyply))
for i, factor in enumerate(pump_mltyply):
    m_lrcQ = copy.deepcopy(lrcQ)
    for sublist in m_lrcQ[1]:
        sublist[-1] *= factor

    wel = flopy.modflow.ModflowWel(mf, stress_period_data=m_lrcQ)
    # Main loop
    if stoch_toggle:
        head_list = []
        conc_list = []
        for n in range(n_real):
            hk = matrix_dict['hk'].copy()
            vk = matrix_dict['vk'].copy()
            x = np.linspace(0, Lx, nrow)
            y = np.linspace(0, Ly, ncol)
        
            for i, layer in enumerate(stoch_lay):
                model = gs.Gaussian(dim=2, var=var[i], len_scale=len_scale)
                srf = gs.SRF(model, seed=10170520 + n + i * 200)
                field = srf.structured([x, y])
                hk[layer, :, :] = hk[layer, :, :] * np.exp(field)
                vk[layer, :, :] = vk[layer, :, :] * np.exp(field)
        
            lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vk, ipakcb=53)
            mf.write_input()
            success, buff = mf.run_model()
            if not success:
                raise Exception("MODFLOW did not terminate normally.")
        
            hds = bf.HeadFile(f"{modelname}.hds")
            
            if n == 0:
                times = hds.get_times()
            head = hds.get_data(totim=times[-1])
            head_list.append(head)
            
            mt.write_input()
            suc6, buff2 = mt.run_model()
            
            ucnobj = bf.UcnFile('MT3D001.UCN')
            conc = ucnobj.get_data(totim=times[-1])
            conc_list.append(conc)
    
        head_stack = np.stack(head_list)
        average_head = np.mean(head_stack, axis=0)
        
        conc_stack = np.stack(conc_list)
        average_conc = np.mean(conc_stack, axis=0)
        
        # Print the shape of the result to verify
        print("Shape of the average matrix:", average_head.shape)
        
        head = average_head
        conc = average_conc
        
    else:
        hk = matrix_dict['hk']
        vk = matrix_dict['vk']
        lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vk, ipakcb=53)
        
        mf.write_input()
        success, buff = mf.run_model()
        if not success:
            raise Exception("MODFLOW did not terminate normally.")
            
        mt.write_input()
        suc6, buff2 = mt.run_model()
        
        hds = bf.HeadFile(f"{modelname}.hds")
        times = hds.get_times()
        
        ucnobj = bf.UcnFile('MT3D001.UCN')
    
        # Read the head and concentration files

        cbc = flopy.utils.CellBudgetFile(f"{modelname}.cbc")
        times = ucnobj.get_times()

        flow_lower_face = cbc.get_data(text='FLOW LOWER FACE', totim=times[-1])[0]
        conc_data = ucnobj.get_data(totim=times[-1])
        
        # Initialize flux array
        flux = np.zeros(flow_lower_face.shape)
        
        # Loop through layers and calculate flux
        for k in range(flow_lower_face.shape[0] - 1):
            flux[k, :, :] = flow_lower_face[k, :, :] * (conc_data[k, :, :] + conc_data[k+1, :, :]) / 2

        pos_flux = np.sum(np.where(flux[8,:,:] > 0, flux[8,:,:], 0))    
        total_flux = np.sum(flux[8,:,:])
        flux_array[i] = total_flux
        posflux_array[i] = pos_flux
    
        # Optionally, plot the flux for visualization
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(flux[7, :, :], cmap='viridis', extent=(0, flux.shape[2], 0, flux.shape[1]))
        plt.colorbar(label='Contaminant Flux')
        plt.title(f'Contaminant Flux at Time {times[-1]}, for pump factor {factor}')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()

print(total_flux)

#%%
fontsize = 9
print(flux_array)
plt.figure(dpi=300)
plt.plot(pump_mltyply, flux_array, '-o', color = 'blue')
plt.xlabel("Pumping rate multiplication factor", fontsize = fontsize)
plt.ylabel("Total nett flux from $8^{th}$ to $9^{th}$ layer [$kg\,m^{-3} d^{-1}$]", fontsize = fontsize)
plt.show()

fontsize = 9
print(flux_array)
plt.figure(dpi=300)
plt.plot(pump_mltyply, posflux_array, '-o', color = 'blue')
plt.xlabel("Pumping rate multiplication factor", fontsize = fontsize)
plt.ylabel("Total flux from $8^{th}$ to $9^{th}$ layer [$kg\,m^{-3} d^{-1}$]", fontsize = fontsize)
plt.show()