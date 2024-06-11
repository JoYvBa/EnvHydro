# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:41:31 2024

@author: Jorrit
"""

import numpy as np
import flopy

def read(file, x, y):
    '''
    Reads in .txt files in the format of the ASCII tables exported from PMWIN
    and converts them to a 2D numpy array.

    Parameters
    ----------
    file : String
        File path to a .txt file exported from PMWIN.
    x : Integer
        Amount of cells in the x direction of the model (amount of columns).
    y : Integer
        Amount of cells in the y direction of the model (amount of rows).

    Returns
    -------
    matrix : 2D-numpy array
        Values for a certain parameter in the x and y direction, structured in a numpy array.

    '''
    # Open the .txt file
    with open(file, 'r') as table:
        # Read the .txt file and save it as a list. The .txt file is delimited by a varying amount of spaces
        # depending on the size of the values. Just above half of these spaces is replaced by a comma,
        # after which all spaces and enters are deleted. This is then made into a list by splitting
        # on commas. As the first entries in the .txt file are the x and y extents and no longer needed,
        # they are deleted.
        data = table.read().replace("       ", ",").replace(' ', "").replace("\n", "").split(",")[2:]
        # Make an array of the list of parameter values
        darray = np.array(data)
        # Shape the array to the dimensions/extent of the model
        matrix = darray.reshape(y, x)
        
    return matrix

def hfbconvert(hfb, k_hfb):
    '''
    Converts a 3D array with values indicating which face of a cell contains a 
    horizontal flow barrier (hfb) and converts it to a list which indicates between 
    which two cells a barrier is located. The latter is the input format for
    hfb in FloPy.

    Parameters
    ----------
    hfb : 3D numpy array 
        3D array with values representing face of a cell where a hfb is present.
        0 is no hfb, 1 is hfb on the left face, 2 is hfb on the right face, 3 is
        hfb on the top face, 4 is hfb on the bottom face. It is identical to the
        .txt export format of the hfb from PMWIN.
    k_hfb : float
        The hydraulic conductivity of the hfb in m/d

    Returns
    -------
    hfb_data : List
        A list of lists representing the cells between which a hfb is present,
        and its hydraulic conductivity. The format for the inner list is:
        [layer, row1, col1, row2, col2, k_hfb].

    '''
    # Make an empty list to save the hfb in
    hfb_data = []
    
    # Make a list of containing the layer, row and column of each cell that has an adjacent hfb
    a = np.asarray(np.where(hfb != 0)).T.tolist()

    # For each of those cells:
    for cell in a:
        # Save the layer, row and column
        layer = cell[0]
        row1 = cell[1]
        col1 = cell[2]
        # If the cell has the value 1...
        if hfb[tuple(cell)] == 1:
            # Then the hfb is to the left, thus the other adjacent cell would be a column left of this cell.
            row2 = row1
            col2 = col1 - 1
        # If the cell has the value 2...
        elif hfb[tuple(cell)] == 2:
            # Then the hfb is to the right, thus the other adjacent cell would be a column right of this cell.
            row2 = row1
            col2 = col1 + 1
        # If the cell has the value 3...
        elif hfb[tuple(cell)] == 3:
            # Then the hfb is to the top, thus the other adjacent cell would be a row up of this cell.
            row2 = row1 - 1
            col2 = col1
        # If the cell has the value 4...
        elif hfb[tuple(cell)] == 4:
            # Then the hfb is to the btoom, thus the other adjacent cell would be a row down of this cell.
            row2 = row1 + 1
            col2 = col1 
            
        # Combine the values into a list and add them to the main list.
        hfb_data.append([layer, row1, col1, row2, col2, k_hfb])
    
    return hfb_data

# Will not be included in the final version, if it is still there; oops
def validate_hfb_data(hfb_data, nlay, nrow, ncol):
    for entry in hfb_data:
        layer, row1, col1, row2, col2, hfb = entry
        if not (0 <= layer < nlay and 0 <= row1 < nrow and 0 <= col1 < ncol and 0 <= row2 < nrow and 0 <= col2 < ncol):
            print(f"Out-of-bounds entry found: {entry}")
            return False
    return True


def ghbconvert(ghb, k_ghb):
    '''
    Converts a 3D array with values for the conductivity of the head and conductivity
    of the general head boundary (ghb) and converts it to a list which contains
    the layer, row and column on which the ghb is present. This is the input format
    for ghb FloPy.

    Parameters
    ----------
    ghb : 3D numpy array
        3D array with the head value of the ghb across the model extent. For cells
        without a ghb, the value is 0.
    k_ghb : float
        The conductivity of the general head boundary, calculated as: *insert calc*

    Returns
    -------
    ghb_data : List
        List of lists representing the cells with a general
        head boundary, its head and conductivity, for a single stress period. The 
        format of the inner list is the following [layer, row, column, h_ghb, k_ghb].

    '''
    
    # Make a dictionary with one list for a stress period (this model does not change
    # the ghb for different stress periods, thus only 1 is used here.)
    ghb_data = {0: []}
    
    # Store the layer, row and column of each cell with a general head boundary in a list
    b = np.asarray(np.where(ghb != 0)).T.tolist()

    # For each of those cells:
    for cell in b:
        # Save layer, row, column, head and conductivity to the list in the dictionary.
        ghb_data[0].append([cell[0], cell[1], cell[2], ghb[tuple(cell)], k_ghb])
    
    return ghb_data

def s_period_con_convert(spd, itype):
    
    itype = flopy.mt3d.Mt3dSsm.itype_dict()[itype]
    stress_period_data = []
    
    c = np.asarray(np.where(spd !=0)).T.tolist()
    
    for cell in c:
        stress_period_data.append([cell[0], cell[1], cell[2], spd[tuple(cell)], itype])
    
    return stress_period_data
    