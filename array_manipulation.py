
import numpy as np


def transform_data(arr, indices):
    return arr[:,:,indices]


"""
INDICES OF INTEREST : 
Fp1,Fp2,F7,F3,Fz,F4,F8,FC3,FCz,FC4,T7,C3,Cz,C4,T8,CP3,CPz,CP4,P7,P3,Pz,P4,P8,O1,Oz,O2
 0   1  2  3  4  5  6  7   8   9   10 11 12 13 14 15  16  17  18 19 20 21 22 23 24 25 
"""

# indices_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 20]
