#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:16:34 2026

@author: Seamus Anderson
"""

import crater_pads_LIB as LIB
import numpy as np
from datetime import datetime as dtm


     # DEMO #
     ########        
if(True):
    t0 = dtm.now()
    y_n = input('Would you like to Run the Demo? (Y/N)\n')
    
    if(y_n == 'Y' or
       y_n == 'y'):
        print('Starting Demo...')
        
            # Speeds in [m/s]
        speed_low  =   10
        speed_high = 2380
        speed_intv =   10
        
            # ejecta angles in [deg]
        deg_low   =  0
        deg_high  = 20
        deg_intv  =  0.1
            
            #Init particle velocities
        v0s  = np.arange(speed_low, speed_high + speed_intv, speed_intv)
        degs = np.arange(deg_low,   deg_high   + deg_intv,   deg_intv)
            # Designate DTM img dir
        dir_fname = 'demo/demo_NAC_DTM_AITKEN01'
            # Make crater Profiles
        LIB.get_crater_profiles(dir_fname, plot=True, use_YOLO=True, floor_slope_lim=6, plotting_width=1.5)
            # Run simulation on craters
        LIB.simulate_PSIs_on_DTM(dir_fname, v0s, degs)    

    else:
        print('Answer was no, terminating...')


    print('Elapsed Time: ', dtm.now() - t0)
    
    # On 20 March 2026
    #  Elapsed Time: 23 min with 20 threads on Intel i9-12900 w/ 64 GB on Ubuntu 24.04.3 LTS
    
    