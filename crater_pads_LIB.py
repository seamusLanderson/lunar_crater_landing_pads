# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:23:56 2024

@author: Seamus Anderson

"""
from datetime import datetime as dtm
from pprint import pprint
from sys import getsizeof

#import crater_pads_IMGLIB as IMGLIB
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import numpy as np

import mpl_toolkits
import inspect
import shutil
import pickle
import scipy
import copy 
import time 
import PIL
import os
import gc

import tkinter as tk
import tkinter.filedialog

from scipy import stats
from scipy.ndimage import gaussian_filter1d


PIL.Image.MAX_IMAGE_PIXELS = None


    # Lunar Gravity
global g
g = 1.625           # [m/s**2]

LRV_pitch_lim = 25  # [deg]


    # Check number of computer cores
global colors
    # Okabe & Ito color pallete
colors = ( 
          (0,     0,   0),      # black
          (.90, .60,  0),       # orange
          (.35, .70, .90),      # light blue
          (.0,     .60, .50),   # blue green
          (.95, .90, .25),      # yellow
          (0,     .45, .70),    # blue
          (.80, .40, 0),        # dark orange
          (0,     .60, .70),    # red purple
          (0,     0,   0),      # black
          (.90, .60,  0),       # orange
          (.35, .70, .90),      # light blue
          (.0,     .60, .50),   # blue green
          (.95, .90, .25),      # yellow
          (0,     .45, .70),    # blue
          (.80, .40, 0),        # dark orange
          (0,     .60, .70) )   # red purple

global markers
markers = ['o', '^', 's', '*', 'P', 'd', 'X',
           '^', 's', '*', 'P', 'd', 'X']





#### Custom Objects ####
########################

    # Define crater_profile class
class Crater_Profile(object):
    def __init__(self, x, y, theta, width, depth, ID_string, max_theta, l_rim, r_rim):
        self.x         = x              # [m]
        self.y         = y              # [m]
        self.theta     = theta          # [deg]
        self.width     = width          # crater width [m]
        self.depth     = depth          # crater depth [m]
        self.ID_string = ID_string      # ID string for crater startX_startY_Width_Height
        self.max_theta = max_theta      # maximum slope within crater
        self.l_rim     = l_rim          # left crater rim relative to crater center
        self.r_rim     = r_rim          # right crater rim relative to crater center

    # Define Particle_track class
class Particle_track(object):
    def __init__(self, vx0, vy0, impact_x, impact_y, impact_slope, impact_theta, impact_v, img, x, y):
        self.vx0          = vx0            # init x velocity [m/s]
        self.vy0          = vy0            # init y velocity [m/s]
        self.impact_x     = impact_x       # impact position x [m]
        self.impact_y     = impact_y       # impact position y [m]
        self.impact_slope = impact_slope   # crater slope at impact point [deg]
        self.impact_theta = impact_theta   # impact angle [deg]
        self.impact_v     = impact_v       # impact speed [m/s]
        self.img          = img            # Unused image pf particle track for heat map display
        self.x            = x              # x position array [m]
        self.y            = y              # y position array [m]
        
    # Define particle swarm save object
class Particle_Swarm(object):
    def __init__(self, init_vx, init_vy, impact_x, impact_y, impact_slope, impact_angle, shadow_x, shadow_y, heat_img, impact_v):
        self.init_vx      = init_vx
        self.init_vy      = init_vy
        self.impact_x     = impact_x
        self.impact_y     = impact_y
        self.impact_slope = impact_slope
        self.impact_angle = impact_angle
        self.shadow_x     = shadow_x
        self.shadow_y     = shadow_y
        self.heat_img     = heat_img
        self.impact_v     = impact_v
        
########################



#################### Functions ########################
#######################################################
#######################################################
#######################################################


def list_files(folder, f_type='.jpg', exclude=None):
	'''Takes a folder and file extention, and returns all the filenames (fullpaths) in that
		   directory with that extention.
		   
	   ** NOTE this can work for any sub-dir or file if you replace the extention with what **
		* you want to look for							  *
	
	Inputs:
			folder  [str]   full path to the folder to be searched
			f_type  [str]   file extention including the '.'
			exclude [str]   key char/string to exclude when looking for files
			
	Outputs
			files [list]	 full path to files containing 'f_type' extention, sorted
	'''
		
		# List contents of the given 'folder'
	contents = os.listdir(folder)
		# Prep list for files to be extracted
	files = []
		# Loops through all the items found in 'folder'
		#  adds them to the return list 'files', if file extention matches
	for i in range(len(contents)):
		if(f_type.upper()  in contents[i].upper()):
			files.append(os.path.join(folder, contents[i]))
		if((exclude != None) and (exclude.upper() in contents[i].upper())):
			files.remove(os.path.join(folder, contents[i]))

	return sorted(files)




#######################################################
#######################################################
#######################################################


def get_v0(Dp, x, Lh, Lm):
    ''' Get ejection velocity of regolith particle due to PSI from Fontes et al. 2022
    ********** Currently Not Used *************    
    
    
    '''
    
        # From Fontes et al. 2022 (Acta Astro.)
    
    # Dp : [m]  particel diam 
    # x  : [m]  radial starting position
    # Lh : [m]  Lander height
    # Lm : [metric tons] Lander mass

    # Calculate magnitude of velocity (starting)
    
    	# From Email with Doug Fontes
    c1 = 0.400
    c2 = 1.362
    c3 = 0.415
    c4 = 0.695
    c5 = 3.718
    c6 = 5.084
    c7 = 0.62
    c8 = 1.846
    c9 = 0.770

    coeff = (Lm**c1) * (Lh**(-c2)) * (Dp**(-c3))
    ex1   = -(x - (Lh     - c5))**2 * (2.*(c6**2))**-1
    ex2   = -(x - (Lh**c7 - c8))**2 * (2.*(c9**2))**-1


    v0 = coeff * ((c4 * Lh * np.exp(ex1)) + np.exp(ex2))


    return v0


#######################################################
#######################################################
#######################################################


def calc_crater_wall(x, crater_profile):
    ''' Calculates the height and slope of any point along a crater profile. Usually heights are known for every 2-5 m, 
    if you give this a point at 1.284 m (x position) it will give you the height at that position and and the slope.
    
    Inputs:
            x              [float] x position (in meters) where you would like to calculate the height and slope
            crater_profile [obj]   the crater profile object you are using
    
    Outputs:
            y     [float] height of the crater (in m)  at the given x position
            slope [float] slope of the crater (in deg) at the given x position
        
    
    '''

    
        # x in [m]
    cx = crater_profile.x
    cy = crater_profile.y
    ct = crater_profile.theta
   
    # Find the known heights to the right and left of your x coordinate
    lessthan = np.where(x >  cx)[0]
    morethan = np.where(x <  cx)[0]
    matches  = np.where(x == cx)[0]

        # Check if x coordinate is already in know elevation points
    if(len(matches) > 0):
        if(matches[0] == 0):
            return cy[matches[0]], ct[matches[0]]
		
        else:
            return cy[matches[0]], ct[matches[-1]]
    
        # Check if supplied x coordinate is out of bounds, if yes return last known crater y val
        	# Too low
    if(len(lessthan) < 1):
        return cy[0],  0
    
    		# Too high
    if(len(morethan) < 1):

        return cy[-1], 0
        
        
    x0_index = lessthan[-1]
    x1_index = morethan[0]

    x0 = cx[x0_index]
    y0 = cy[x0_index]

    x1 = cx[x1_index]
    y1 = cy[x1_index]

		# out of order check 
    if(x0 > x1):
        x0 = cx[x1_index]
        y0 = cy[x1_index]

        x1 = cx[x0_index]
        y1 = cy[x0_index]
		


		# rise / run
    m = (y1 - y0) / (x1 - x0)

    b = y0 - m * x0


    y = x * m + b
    
    return y, np.rad2deg(np.arctan(abs(m)))
    

#######################################################
#######################################################
#######################################################


def make_artificial_landing_pads(save_dir, wall_heights=np.arange(5,35,5), wall_radii=np.arange(20, 220, 20)):
    ''' ####### NOT USED ############# 
    
    Makes 'crater profiles' which are actually artificial landing pads. 
    Also saves the profiles in .pkl files.
    
    wall_heights [array] in [m]
    wall_radii   [array] in [m]
    '''
    
    if(not os.path.exists(save_dir)):
        os.mkdir(save_dir)

    pkl_snames = []

        # move though each listed wall height and radius
    for i in range(len(wall_heights)):
        for j in range(len(wall_radii)):
            x = []
            y = []
            
                # Add x and y points
            x.append(-200000),             y.append(0)
            x.append(-wall_radii[j] - 2),  y.append(0)
            x.append(-wall_radii[j] - 1),  y.append(wall_heights[i])
            x.append(-wall_radii[j] + 1),  y.append(wall_heights[i])
            x.append(-wall_radii[j] + 2),  y.append(0)
        
            x.append( wall_radii[j] - 2),  y.append(0)
            x.append( wall_radii[j] - 1),  y.append(wall_heights[i])
            x.append( wall_radii[j] + 1),  y.append(wall_heights[i])
            x.append( wall_radii[j] + 2),  y.append(0)
            x.append( 200000),             y.append(0)

            x = np.asarray(x)
            y = np.asarray(y)    
    
    

        
            theta = np.zeros(len(x)-1)
            width = wall_radii[j] * 2
            depth = wall_heights[i]

            ID_string = 'Artificial_Pad_W-' + str(width) + 'H-' + str(depth)
            max_theta = 0             

            this_profile = Crater_Profile(x, y, theta, width, depth, ID_string, max_theta)

                # save to pkl file
            pkl_sname  = os.path.join(save_dir,(ID_string + '.pkl'))
            pkl_snames.append(pkl_sname) 

            with open(pkl_sname, 'wb') as out:
                pickle.dump(this_profile, out, pickle.HIGHEST_PROTOCOL)       
                

                # plot
            plot_sname = os.path.join(save_dir, (ID_string + '.png'))

            txt_str = ('\n Incline: ' + str(max_theta) + ' deg' + 
                       '\n Diam:    ' + str(width) + ' [m]' + 
                       '\n Depth:   ' + str(depth.round(1)) + ' [m]' +
                       '\n d/D:     ' + str((depth/width).round(2)))
            
            plt.plot(x, y)
            plt.text(width, depth+50, txt_str)
            plt.axis('equal')
            plt.ylabel('[m]')
            plt.xlabel('[m]')
            plt.xlim((-2000, 2000))

            plt.savefig(plot_sname)
            plt.clf()

    return pkl_snames


#######################################################
#######################################################
#######################################################


def convert_color_tif(tif_fname):
    ''' Convert a DTM color gradient image to a png for crater labelling
    
    Inputs:
        tif_fname [str] fullpath to the colorgradient.tif
        
    Ouputs:
        [file] writes a png with same imag name as the input .tif
       
    '''
    
    sname = tif_fname.replace('_CLRGRAD.TIF','.png')
    
    IMG = PIL.Image.open(tif_fname)
    IMG.save(sname)
    
    
    return


#######################################################
#######################################################
#######################################################


def get_crater_profiles(dir_fname, kmpix=2, plot=True, use_YOLO=True, floor_slope_lim=6, plotting_width=1.5):
    ''' Make crater profile objects and saves them as .pkl files for later use in simulations. 
    
    Inputs:
        dir_fname       [str]  full path to directory that has .TIF, .LBL, and .txt file with DTM, labelfile, and crater labels, respectively
        kmpix           [int]  Unused, now auto detemrined by accompanying .LBL file
        plot            [bool] If yes then plot profile  
        use_YOLO        [bool] If yes then use YOLO format where .txt annotations has same name as .TIF DTM
        floor_slope_lim [int]  unused
    
    Outputs:
        [files] .pkl files that contain the Crater_Profile object made from .txt annotation file
                made in subdir: 'NAC_DTM_profiles/crater_plots_and_pkl/'
    
    '''
    
    short_name = os.path.split(dir_fname)[1].replace('a_','')
       

        # Open image
    img_fname = os.path.join(dir_fname, short_name + '.TIF')
    png_fname = os.path.join(dir_fname, short_name + '.png')

    IMG_arr = np.asarray(PIL.Image.open(img_fname))
    
        # Read Image-J labels (.csv)        
    if(use_YOLO == False):
        csv_fname = os.path.join(dir_fname, 'Results.csv')
        df = pd.read_csv(csv_fname)

        # If YOLO/Label studio .txt file exists    
    if(use_YOLO == True):
        csv_fname = img_fname.replace('.TIF','.txt') 
        with open(csv_fname, 'r') as f:
            datalines = f.readlines()
        df = pd.DataFrame(columns=['Label', 'BX', 'BY', 'Width', 'Height'])
        
        X, Y, W, H = [], [], [], [],
        Label = [os.path.split(img_fname.replace('.TIF','.png'))[1] for i in range(len(datalines))]
        
        for i in range(len(datalines)):
            line = datalines[i].split(' ')
            
            W.append(int(float(line[3]) * IMG_arr.shape[1]) )
            H.append(int(float(line[4]) * IMG_arr.shape[0]) )                
            
            X.append(int(float(line[1]) * IMG_arr.shape[1]) - int(W[i]/2)  )
            Y.append(int(float(line[2]) * IMG_arr.shape[0]) - int(H[i]/2)  )                
            
        df['Label']  = Label
        df['BX']     = X
        df['BY']     = Y
        df['Width']  = W
        df['Height'] = H
    
    
        # Read lbl file (meta data for image)
    lbl_fname = img_fname.replace('.TIF', '.LBL')
    f = open(lbl_fname, 'r')
    lines = f.readlines()
    f.close()
    	# Set map scale from lbl file
    for i in range(len(lines)):
    	if('MAP_SCALE' in lines[i]):
    		kmpix = np.round(float(lines[i].split('=')[1].split('<')[0]), 0)
    
        # Make save dir
    save_dir = img_fname.replace('.TIF', '_profiles')
    plot_dir = os.path.join(save_dir, 'crater_plots_and_pkl')
    
    if(not os.path.exists(save_dir)):
        os.mkdir(save_dir)
    if(not os.path.exists(plot_dir)):
        os.mkdir(plot_dir)

        # Move through each crater in the .csv dataframe
    n_craters = len(df)
    crater_profile_fnames = []
    crater_profile_list   = []

    for i in range(n_craters):
        	
        # deconstruct df entry
        name = df['Label'].iloc[i]
        X    = df['BX'].iloc[i]
        Y    = df['BY'].iloc[i]
        W    = df['Width'].iloc[i]
        H    = df['Height'].iloc[i]
        
            # Determine crater center          
        cx = np.where(IMG_arr[Y:Y+H, X:X+W] == np.min(IMG_arr[Y:Y+H, X:X+W]))[1] 
        cy = np.where(IMG_arr[Y:Y+H, X:X+W] == np.min(IMG_arr[Y:Y+H, X:X+W]))[0] 
        
            # check if there are multiple minimums, if so hard select center
        if(len(cx) == 1 or
           len(cy) == 1 ):       
            cx = cx[0]
            cy = cy[0]       

        else:        
            cx = W//2
            cy = H//2
            
        cx += X
        cy += Y
        
            # Slice the crater from the image, 
            #  either width-wise or height-wise depending on which axis is longer    
            # Also find crater rims
        if(W >= H):
            cc    = cx #crater center
            y     = IMG_arr[cy,:].flatten()                       
            x     = np.arange(len(y))
            l_rim = (X-cc) * kmpix
            r_rim = (X+W-cc) * kmpix
        if(W < H):
            cc    = cy #crater center
            y     = IMG_arr[:,cx].flatten()
            x     = np.arange(len(y))
            l_rim = (Y-cc)   * kmpix
            r_rim = (Y+H-cc) * kmpix

            
            # Shift the crater up/down so the bottom y = 0
        y -= y[cc]
        
            # Shift the crater left/right so the center x = 0
        x -= x[cc]
        
            # Scale x, width, height to unit [m], y is already in [m]
        x       = x * kmpix
        width   = W * kmpix
        height  = H * kmpix
                
            # Calculate Depth 
        depth = np.mean(y[cc + int(l_rim/kmpix)] + y[cc + int(r_rim/kmpix)]) 
        
            # Calculate max angle within crater 
        dx = np.array([x[j+1]-x[j] for j in range(len(x)-1)])
        dy = np.array([y[j+1]-y[j] for j in range(len(y)-1)])
    
        theta = np.rad2deg(np.arctan(dy/dx))    
        
        
        if(W >= H):
            max_theta = np.max(theta[int(cc-(W*.55)):int(cc+(W*.55))])
        if(W < H):
            max_theta = np.max(theta[int(cc-(H*.55)):int(cc+(H*.55))])    
        
                
            # Clip non usable part of the image
        mask = np.where(y > -2000000)
        y = y[mask]
        x = x[mask]
          
            # Add 200 [km] to either side
        x = np.concatenate([[-100_000], x])
        x = np.concatenate([            x, [100_000]])
		
        y = np.concatenate([[y[0]], y])
        y = np.concatenate([        y, [y[-1]]])
            
            # Recalculate crater angle with clipped edges and added 100 [km]
        dx = np.array([x[j+1]-x[j] for j in range(len(x)-1)])
        dy = np.array([y[j+1]-y[j] for j in range(len(y)-1)])
    
        theta = np.rad2deg(np.arctan(dy/dx)) 
        
            # Calc the crater floor
        longest_floor = 0
        
        if(W >= H):
            floor_index = np.where(theta[int(cc-(W*.50))+1:int(cc+(W*.50))+1] <= floor_slope_lim)
        if(W < H):
            floor_index = np.where(theta[int(cc-(H*.50))+1:int(cc+(H*.50))+1] <= floor_slope_lim)    
            

            # Create crater ID
        ID_string  = 'IMG_' + short_name 
        ID_string += '-X_' + str(X) + '-Y_' + str(Y) + '-W_' + str(W) + '-H_' + str(H) 
            
            # add crater to return list and save
        this_profile = Crater_Profile(x, y, theta, width, depth, ID_string, max_theta, l_rim, r_rim)
 
        
        pkl_sname = os.path.join(plot_dir, (ID_string + '.pkl'))
        crater_profile_fnames.append(pkl_sname)
        crater_profile_list.append(this_profile)
        with open(pkl_sname, 'wb') as out:
            pickle.dump(this_profile, out, pickle.HIGHEST_PROTOCOL)       
                     
                 
            # Prepare slope vals for plotting
        slopes = [calc_crater_wall(pos + kmpix/2, this_profile)[1] for pos in x[1:-1]]         
        
            # Annotate Image for plotting
        

            # Plot this crater
        if(plot == True):
            plot_sname = os.path.join(plot_dir, (ID_string + '.png'))

            txt_str = ('\n Incline:  ' + str(max_theta.round(1)) + ' deg' + 
                       '\n Diam:     ' + str(width) + ' [m]' + 
                       '\n Depth:    ' + str(depth.round(1)) + ' [m]' +
                       '\n Depth/Diam' + str((depth/width).round(2)))
            
            fig, axs = plt.subplots(2)
            
            xlim_coeff = 1
            
            crater_profile = this_profile
            crater_radius  = crater_profile.width/2
            #major_ticks = np.array([crater_profile.l_rim, crater_profile.r_rim])
            major_ticks = np.concatenate( [np.arange(-(crater_radius)*10, -crater_radius, crater_radius),
                                           [crater_profile.l_rim],
                                           [0],
                                           [crater_profile.r_rim],
                                           np.arange(crater_radius*2, crater_radius*10, crater_radius)
                                            ])                           
                
                # Plot crater profile
            axs[0].plot(x, y)
            axs[0].text(width, depth+50, txt_str)
            axs[0].set_xticks(major_ticks)
            axs[0].set_yticks(np.arange(-1000, 1000, 50))
            axs[0].grid()
            axs[0].axis('equal')
            axs[0].set_ylabel('[m]')
            axs[0].set_xlabel('[m]')
            axs[0].set_xlim((-xlim_coeff*width, xlim_coeff*width))
            axs[0].set_ylim((-3*depth, 3*depth))
                # plot crater limits as black dots
            axs[0].plot(l_rim , calc_crater_wall(l_rim, this_profile)[0], 'ko')
            axs[0].plot(r_rim , calc_crater_wall(r_rim, this_profile)[0], 'ko')

                # Plot the slope of the profile                
            axs[1].plot(x[1:-1] + kmpix/2,  slopes, 'ko')
            axs[1].set_ylabel('Slope [$^\circ$]')
            axs[1].set_xlabel('[m]')
            axs[1].set_ylim((-1, 50))
            axs[1].set_xlim((-xlim_coeff*width, xlim_coeff*width))
            axs[1].hlines(y=20, xmin=-xlim_coeff*width, xmax=xlim_coeff*width, color='b')   
            axs[1].hlines(y=6,  xmin=-xlim_coeff*width, xmax=xlim_coeff*width, color='r')

                # Show Image
            #axs[2].imshow()



            plt.savefig(plot_sname)
            plt.show()
            plt.clf()
       
    return crater_profile_fnames


#######################################################
#######################################################
#######################################################


def new_run_particle_sim_array(input_arr, xlim=10_000, ylim=10_000, verbose=False, mp=True):
    ''' Not Used. Experimental function for calculating particel position based on 1m x diemnsion increments, instead
    of a give ms time step (usually 10 ms)
    
    
    
    '''
    


    t_init = dtm.now()    

    vx0		 = input_arr[0]
    vy0		 = input_arr[1]
    x0		 = input_arr[2]
    y0		 = input_arr[3]
    dt		 = input_arr[4]
    seconds  = input_arr[5]
    profile  = input_arr[6]
    ret_list = input_arr[7]
    plot_lim = input_arr[8]


    t_max_pos = (-vy0 + np.sqrt(vy0**2 + 2*g*(y0 - np.min(profile.y) ) ) ) / (2*g)
    t_max_neg = (-vy0 - np.sqrt(vy0**2 + 2*g*(y0 - np.min(profile.y) ) ) ) / (2*-g)

    print(t_max_pos)
    print(t_max_neg)


    x = np.arange(0, xlim, 1)
    t = (x - x0) / vx0
    y = 0.5 * g * t**2 + vy0 * t + y0
    
    vx = np.ones(len(x)) * vx0
    vy = g * t + vy0

    h = [calc_crater_wall(this_x, profile)[0] for this_x in x]

    mask = np.where(y-h > 0)


    x  = x[mask]
    y  = y[mask]
    vx = vx[mask]
    vy = vy[mask]
    
    v        = np.sqrt(vx**2 + vy**2)
    theta    = np.rad2deg(np.tan(vy[-1] / vx[-1]))    
    slope    = calc_crater_wall(x[-1], profile)[1]
    impact_v = v[-1]

    this_track = Particle_track(np.round(vx0,    1).astype(np.float16), 
                                np.round(vy0,    1).astype(np.float16), 
                                np.round(x[-1],    1).astype(np.float16), 
                                False,#np.round(y[-1],    1).astype(np.float16), # impact y
                                np.round(slope,    1).astype(np.float16), # crater slope at impact
                                np.round(theta,    1).astype(np.float16), # impact angle
                                np.round(impact_v, 1).astype(np.float16), # impact speed
                                False, #img,        
                                np.round(x,    1).astype(np.float16), # x position array
                                np.round(y,    1).astype(np.float16)) # y position array
    ret_list.append(this_track)


    print(dtm.now() - t_init)

    return



#######################################################
#######################################################
#######################################################



def run_particle_sim_array(input_arr, xlim=10_000, ylim=10_000, verbose=False, mp=True):
    ''' Calculates the ballistic trajectory for one particle for a given profile
    
    Inputs:
        input_arr [array] particle initial parameters
            vx0		[m/s] init x velocity
            vy0		[m/s] init y velcity
            x0		[m]   init x position
            y0		[m]   init y position
            dt		[s]   time step (usually 10 ms)
            seconds [s]	  unused - number of secods to run simulation 
    
        xlim    [m] when particle x posiiton reaches this, stop simulation
        ylim    [m] when particle y posiiton reaches this, stop simulation
        verbose [bool] if yes print everything, probably not a good idea to say yes to this
        mp      [bool] testing param, if yes then assume multiprocessing function call 

    
    '''

    ''' Setup simulation params and crater profile '''
    '''============================================'''
    
    t_init = dtm.now()
    
        # unpack the input array
    vx0		 = input_arr[0]
    vy0		 = input_arr[1]
    x0		 = input_arr[2]
    y0		 = input_arr[3]
    dt		 = input_arr[4]
    seconds  = input_arr[5]
    profile  = input_arr[6]
    ret_list = input_arr[7]
    plot_lim = input_arr[8]

    v = np.sqrt(vx0**2 + vy0**2)    
    
    xlim = profile.width * 2

    dt = 0.01 #[sec]

        # auto adjust height based on local crater height
    y0 += calc_crater_wall(x0, profile)[0]

        # check particle height at crater wall
    t_wall  = (profile.width/2. - x0) / vx0
    y_wall  = y0 + (vy0 * t_wall) + (0.5 * (-g) * t_wall**2)
    
    t_floor = (2 * vy0) / g
        
        # set simulation time to the time it takes to hit the crater floor again
    seconds = t_floor
    
    
        # Doesn't work because apparently I can't solve a quadratic polynomial
    '''
        # Calc time to hit lowest point in profile (within 1.5 crater radii)
    del_y    = np.min(profile.y[np.where(abs(profile.x) < profile.width * 1.5)])
    #t_lowest = (-vy0 + np.sqrt(vy0**2 + (del_y *g) /2. )) / g
    t_lowest = (-vy0 + np.sqrt(vy0**2 + (2*g*(del_y-1)) )) / g
    
        # auto set simulation time to run until particle reaches lowest point in relevant area
    seconds = t_lowest
    '''
    
        # if particle doesn't make it out of the crater
    if(y_wall < calc_crater_wall(profile.width/2., profile)[0]):
        seconds = t_wall
            # check if it hits the floor before the wall 
        if(t_floor < t_wall):
            seconds = t_floor
    
    
    
	    # Initialize track arrays
    step_arr = np.arange(0, seconds + dt, dt)
    n_steps  = len(step_arr)

    
    ''' Run the simulation '''
    '''===================='''
	    # Calculate Ballistics
    t0 = dtm.now()
    
    x  = x0 + (vx0 * step_arr)
    y  = y0 + (vy0 * step_arr) + ((-g * step_arr**2)/2.)
    vx = vx0 * np.ones(n_steps)
    vy = vy0 + (-g * step_arr)

        # find where the particle actually hits the ground
            # Within area of interest
    if(x[-1] <= plot_lim*2):    
        crater_h = [calc_crater_wall(this_x, profile)[0] for this_x in x]
        
        under_mask = np.where(y <= crater_h)
    
            # particle hits the ground
        if(len(under_mask[0]) != 0):
            i = under_mask[0][0]
            
            # particle is still moving, extend simulation by 2
        if(len(under_mask[0]) == 0):
            step_arr = np.arange(0, 2*seconds + dt, dt)
            n_steps  = len(step_arr)

            x  = x0 + (vx0 * step_arr)
            y  = y0 + (vy0 * step_arr) + ((-g * step_arr**2)/2.)
            vx = vx0 * np.ones(n_steps)
            vy = vy0 + (-g * step_arr)            
            
            crater_h   = [calc_crater_wall(this_x, profile)[0] for this_x in x]            
            under_mask = np.where(y <= crater_h)    
            
                # Check again, if its still moving just end where it is
            if(len(under_mask[0]) != 0):
                i = under_mask[0][0]
            if(len(under_mask[0]) == 0):
                i = len(crater_h) - 1
            
            
            # Outside area of interest
    if(x[-1] > plot_lim*2):
        relevant_mask = np.where(x < plot_lim*2)
        relevant_y    = y[relevant_mask]
        relevant_x    = x[relevant_mask]
        
        crater_h = [calc_crater_wall(this_x, profile)[0] for this_x in relevant_x]
        under_mask = np.where(relevant_y <= crater_h)
        
        if(len(under_mask[0]) != 0):
            i = under_mask[0][0]
        if(len(under_mask[0]) == 0):
            i = len(crater_h) - 1
   
    if(verbose == True):
        print('N Steps:\n\t', n_steps)
        print('Under Mask:\n\t', under_mask)
        print('Plot Lim:\n\t', plot_lim)
        
    x  = x[ :i]
    y  = y[ :i]
    vx = vx[:i]
    vy = vy[:i]

    '''
    img = np.zeros((int(plot_lim), int(plot_lim)), dtype=bool)
    for i in range(len(x)):
        if((int(y[i]) < plot_lim-1) and
           (int(x[i]) < plot_lim-1)):
            img[int(y[i]), int(x[i])] += 1
        else:
            break
    '''

    if(len(x) == 0):
        return []
        
    slope      = calc_crater_wall(x[-1], profile)[1]
    theta      = np.rad2deg(np.tan(vy[-1] / vx[-1]))
    impact_v   = np.sqrt(vx[-1]**2 + vy[-1]**2)
    
    ret_mask = np.where(abs(x) < plot_lim+1)
    ret_x = x[ret_mask]
    ret_y = y[ret_mask]
    
    this_track = Particle_track(np.round(vx[0],    1).astype(np.float16), 
                                np.round(vy[0],    1).astype(np.float16), 
                                np.round(x[-1],    1).astype(np.float16), 
                                False,#np.round(y[-1],    1).astype(np.float16), 
                                np.round(slope,    1).astype(np.float16), 
                                np.round(theta,    1).astype(np.float16), 
                                np.round(impact_v, 1).astype(np.float16), 
                                False, #img,        
                                np.round(ret_x,    1).astype(np.float16), 
                                np.round(ret_y,    1).astype(np.float16))
    ret_list.append(this_track)

    if(mp==False):
        return [vx, vy, x, y, theta]
        
    

    #print(dtm.now() - t_init)    

    return #[vx, vy, x, y, theta]




#######################################################
#######################################################
#######################################################


def new_simulate_PSI_with_crater(crater_profile_fname, v0s, degs, x0s=[0], y0s=[1], dt=0.01, seconds=10, n_free_cores=4, plotting_width=1.5, plot=True, old_tracks_img=False):
    ''' Experimental version that calculates particle location based on x intervals not time intervals. 
    ############## NOT USED CURRENTLY #################
    
    Simulate a plume surface interaction using a given crater profile
    and a list of initial starting parameters. All units in meters, seconds, and m/s
        
    Inputs:
        crater_profile_fname [str]   fullpath to the .pkl file with the crater profile
        v0s                  [array] array of initial velocity magnitudes
        degs                 [array] array of initial velocity angles (from horizontal)
        x0s                  [array] array of init x positions
        y0s                  [array] array of init y positions
        dt                   [float] time step [s] for ballistic trajectory
        seconds              [float] unused - number of seconds to run simulation
        n_free_cores         [int]   number of cores to leave free, worst case scenario, this function only uses one core
        plotting_width       [float] limit for plotting (-xlim,xlim) = (-crater.width * plotting_width, crater.width * plotting_width)
        plot                 [bool]  if True, plot results
        old_tracks_img       [bool]  if True, save the particle tracks image the onld way (normalized to 255), probably best to use False
    
    
    '''
        # Determine number of cores to use
    n_cores = os.cpu_count() - n_free_cores
    if(n_cores <= 0):
        n_cores = 1
    #print('Number of Cores To Use:\n\t', str(n_cores))
    
        # 
    short_crater_name = os.path.split(crater_profile_fname)[1]
    t0 = dtm.now() 
    
    ''' Open crater and plot '''
    '''======================'''
    with open(crater_profile_fname, 'rb') as inp:    
        crater_profile = pickle.load(inp)
    
	    # get crater info
    x      = crater_profile.x
    y      = crater_profile.y
    theta  = crater_profile.theta
    width  = crater_profile.width
    depth  = crater_profile.depth
    maxdeg = crater_profile.max_theta
    l_rim  = crater_profile.l_rim
    r_rim  = crater_profile.r_rim
    
        # Find safe landing Zone
    '''
    within_crater = np.where(abs(x) < width*0.38 )
    safe_list     = find_safe_landing(x[within_crater], theta[within_crater], max_slope=6) # [safe_total, start, end]
    safe_total    = abs(abs(safe_list[1]) - abs(safe_list[2])) 
    safe_center   = (safe_list[1] + safe_list[2]) /2

    if(safe_total >= 10):
        x -= safe_center
        crater_profile.x = x
    '''
    
    print('\n\n')
    print('===============')
    print('Target Crater: \n\t', short_crater_name)
    print('               \tCrater Diam:', crater_profile.width )
    print('===============')
    
    crater_profile_neg = copy.deepcopy(crater_profile)

    label_str = ('\nMax Slope: ' + str(np.round(maxdeg, 3)) + ' [$\degree$]' + 
               '\nDiameter: ' + str(width) + ' [m]' +
               '\nDepth: ' + str(np.round(depth, 1)) + ' [m]' +
               '\nd/D: ' + str(np.round(depth/width, 3)) +
               '\nTotal Incline: '  + str(np.round(np.rad2deg(np.arctan(2*depth/(width))), 3) ))
    		     
    rev_x     = np.asarray(list(reversed(x))) * -1
    rev_y     = np.asarray(list(reversed(y)))
    rev_theta = np.asarray(list(reversed(theta)))
    
    rev_crater_profile = Crater_Profile(rev_x, rev_y, theta, width, depth, 'rev', maxdeg, -1*r_rim, abs(l_rim))


    ''' Collect Particles and Prep saving '''
    '''==================================='''
    ret_list     = mp.Manager().list() 
    rev_ret_list = mp.Manager().list()
    particle_list     = []
    rev_particle_list = []

    for y0 in y0s:
        for x0 in x0s:
            for v0 in v0s:
    	        for deg in degs:
    		        vx0 = v0 * np.cos(np.deg2rad(deg))
    		        vy0 = v0 * np.sin(np.deg2rad(deg))
    		        particle_list.append(    [vx0, vy0, x0, y0, dt, seconds,     crater_profile,     ret_list, width*plotting_width])
    		        rev_particle_list.append([vx0, vy0, x0, y0, dt, seconds, rev_crater_profile, rev_ret_list, width*plotting_width])
    

    
    
    print('N particles:\n\t', len(particle_list)*2)

        # list directory structure
    crater_pkl_dir = os.path.split(crater_profile_fname)[0]
    profile_dir    = os.path.split(crater_pkl_dir)[0]
    
        # make track dir and file save names
    track_pkl_dir = os.path.join(profile_dir, 'track_plots_and_pkl')
    if(not os.path.exists(track_pkl_dir)):
        os.mkdir(track_pkl_dir)
    
    track_pkl_sname  = os.path.join(track_pkl_dir, ('tracks_' + short_crater_name))
    track_plot_sname = track_pkl_sname.replace('.pkl','.png')
    
        # Make hillshade
    '''
    hill_fname = os.path.join(track_pkl_dir, 'hillshade_' + ID_str) 
    hill_arr, ID_str = IMGLIB.grab_crater_hillshade(crater_profile_fname, pad_coeff=2)
    HILL_IMG = PIL.Image.fromarray(hill_arr)
    HILL_IMG.save(hill_fname)
    '''
    
        # Prep the lists for eventual saving
    hist_impact_x = []
    hist_impact_v = []

    save_init_vx0 = []
    save_init_vy0 = []
    save_impact_x = []
    save_impact_y = []
    save_impact_v = []
    save_impact_slope = []
    save_impact_theta = []
    
    shadow_bins_pos = np.arange(0, plotting_width*width, 1)
    shadow_bins_neg = np.arange(0, plotting_width*width, 1) * -1
    lowest_ejecta_pos = [[] for bins in shadow_bins_pos]
    lowest_ejecta_neg = [[] for bins in shadow_bins_neg]
    heat_img_pos = np.zeros((int(width*plotting_width), int(width*plotting_width)))
    heat_img_neg = np.zeros((int(width*plotting_width), int(width*plotting_width)))
    
    ''' Run Simulation and Collect info from tracks '''
    '''============================================='''
    
        #################################
        # Simulate particles to the right
    work_pool = mp.Pool(processes=n_cores)
    work_pool.map(run_particle_sim_array, particle_list)
    
    time.sleep(1)

        # Go through each track and plot
    k = 0
    for track in ret_list:
        k += 1
        
        if(k%50 == 0):
            print('\tCalculating Right Side...', str(k), ' / ', str(len(ret_list)), end='\r')
            # record other save params
        save_init_vx0.append(     track.vx0)
        save_init_vy0.append(     track.vy0)
        save_impact_x.append(     track.impact_x)
        #save_impact_y.append(     track.impact_y)
        save_impact_v.append(     track.impact_v)
        save_impact_slope.append( track.impact_slope)
        save_impact_theta.append( track.impact_theta)
                
        this_impact = track.impact_x
        
            # if the track impacted within the area of interest (crater_width * the plotting width)
            #  record the impact site for later histogram
            #  and note the lowest track hewight for an x bin
        if(len(track.x) > 1):
            hist_impact_x.append(track.impact_x)
            hist_impact_v.append(track.impact_v)    
    
                # record track on heatmap
            track_img = np.zeros((int(width*plotting_width), int(width*plotting_width) ), dtype=bool)
            for i in range(len(track.x)):
                if((int(track.y[i]) < width*plotting_width-1) and
                   (int(track.x[i]) < width*plotting_width-1)):
                    track_img[int(track.y[i] + (width*plotting_width*0.5)), int(track.x[i])] += 1
            
            heat_img_pos += track_img
   
                # Go through each of the shadow bins
            for j in range(len(shadow_bins_pos)):
                mask = np.where((track.x >= j) & (track.x <= j+1))
                if(len(mask[0]) >  0):
                    lowest_ejecta_pos[j].append(np.min(track.y[mask]))
            track = []
                    
        else:
            #print('No Return:', track) 
            continue

    
    ################################
        # Simulate particles to the left
    work_pool = mp.Pool(processes=n_cores)
    work_pool.map(run_particle_sim_array, rev_particle_list)

    time.sleep(1)
    
        #Go through each left track and plot
    k = 0
    for track in rev_ret_list:
        k += 1
        if(k%50 == 0):
            print('\t Calculating Left Side...', str(k), ' / ', str(len(rev_ret_list)), end='\r')
            # record other save params
        save_init_vx0.append(     -track.vx0)
        save_init_vy0.append(      track.vy0)
        save_impact_x.append(     -track.impact_x)
        #save_impact_y.append(      track.impact_y)
        save_impact_v.append(     -track.impact_v)
        save_impact_slope.append( -track.impact_slope)
        save_impact_theta.append( -track.impact_theta)
                
        this_impact = -track.impact_x
        
            # if the track impacted within the area of interest (crater_width * the plotting width)
            #  record the impact site for later histogram
            #  and note the lowest track hewight for an x bin
        if(len(track.x) > 1):
            hist_impact_x.append(-track.impact_x)
            hist_impact_v.append(-track.impact_v)    
    
                # record track on heatmap
            track_img = np.zeros((int(width*plotting_width), int(width*plotting_width) ), dtype=bool)
            for i in range(len(track.x)):
                if((int(track.y[i]) < width*plotting_width-1) and
                   (int(track.x[i]) < width*plotting_width-1)):
                    track_img[int(track.y[i] + (width*plotting_width*0.5)), int(track.x[i])] += 1
            
            heat_img_neg += track_img
            
                # Go through each of the shadow bins
            for j in range(len(shadow_bins_neg)):
                mask = np.where((track.x >= j) & (track.x <= j+1))
                if(len(mask[0]) >  0):
                    lowest_ejecta_neg[j].append(np.min(track.y[mask]))
            track = []
            
        else:
            #print('No Return:', track) 
            continue

    
    sim_t  = dtm.now()
    sim_dt = sim_t - t0
    print('Simulation Time:\n\t', sim_dt)
    
    
    ''' Compile Gap Info '''
    '''=================='''
        # Compile lowest ejecta        
    for j in range(len(lowest_ejecta_pos)):
        if(len(lowest_ejecta_pos[j]) == 0):
            lowest_ejecta_pos[j].append(width*plotting_width)
        lowest_ejecta_pos[j] = np.min(lowest_ejecta_pos[j])
    lowest_ejecta_pos = np.asarray(lowest_ejecta_pos)

    for j in range(len(lowest_ejecta_neg)):
        if(len(lowest_ejecta_neg[j]) == 0):
            lowest_ejecta_neg[j].append(width*plotting_width)
        lowest_ejecta_neg[j] = np.min(lowest_ejecta_neg[j])
    lowest_ejecta_neg = np.asarray(lowest_ejecta_neg)

    shadow_bins_total   = np.concat([list(reversed(shadow_bins_neg)),   shadow_bins_pos])
    lowest_ejecta_total = np.concat([list(reversed(lowest_ejecta_neg)), lowest_ejecta_pos])

    crater_floor  = np.asarray([calc_crater_wall(x + 0.5, crater_profile)[0] for x in shadow_bins_total])    
    gaps          = lowest_ejecta_total - crater_floor


    ''' Compile Heat Imgs '''
    '''==================='''
    if(old_tracks_img==True):
                # Pos
        heat_img_pos  = heat_img_pos / np.max(heat_img_pos)
        heat_img_pos *= 3/4
        heat_img_pos += 1/4
        heat_img_pos = np.where(heat_img_pos == np.min(heat_img_pos), 0, heat_img_pos)
    
        wht_img_pos  = np.ones((int(width*plotting_width), int(width*plotting_width), 3))
        wht_img_pos[:,:,0] -= heat_img_pos
        wht_img_pos[:,:,1] -= heat_img_pos
    
                # Neg
        heat_img_neg  = heat_img_neg / np.max(heat_img_neg) #########################################################
        heat_img_neg *= 3/4
        heat_img_neg += 1/4
        heat_img_neg = np.where(heat_img_neg == np.min(heat_img_neg), 0, heat_img_neg)
    
        wht_img_neg  = np.ones((int(width*plotting_width), int(width*plotting_width), 3))
        wht_img_neg[:,:,0] -= heat_img_neg
        wht_img_neg[:,:,1] -= heat_img_neg
    
            # Total
        IMG1 = PIL.Image.fromarray((wht_img_pos*255).astype(np.uint8))
        IMG2 = PIL.Image.fromarray((wht_img_neg*255).astype(np.uint8))
        IMG2 = IMG2.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
            
        final_IMG = PIL.Image.new('RGB', (wht_img_pos.shape[1]*2, wht_img_pos.shape[0]))
        final_IMG.paste(IMG2, (0,                    0))
        final_IMG.paste(IMG1, (wht_img_pos.shape[1], 0))
        
        final_img_arr = np.asarray(final_IMG) / 255.

    if(old_tracks_img != True):
        final_img_arr = np.zeros((heat_img_pos.shape[0], heat_img_pos.shape[1]*2))
        final_img_arr[:, 0:heat_img_pos.shape[1]]  = np.flip(heat_img_neg, axis=1) 
        final_img_arr[:,   heat_img_pos.shape[1]:] = heat_img_pos                 
        

        

    ''' Save tracks info '''
    '''=================='''
    file_obj = open(track_pkl_sname, 'wb')    

    short_mask   = np.where(abs(x) < width*plotting_width)
    short_crater = Crater_Profile(x[short_mask], y[short_mask], theta[short_mask], width, depth, crater_profile.ID_string, crater_profile.max_theta, l_rim, r_rim)
    
    swarm = Particle_Swarm(
    
                   save_init_vx0, 
                   save_init_vy0,
                   save_impact_x,
                   save_impact_y,
                   save_impact_slope,
                   save_impact_theta,
                   shadow_bins_total,
                   gaps,
                   final_img_arr,
                   save_impact_v)    
    
    pickle.dump(swarm,          file_obj)
    pickle.dump(short_crater, file_obj)
    
    file_obj.close()
   
    save_dt = dtm.now() - sim_t
    print('Save time:\n\t', save_dt)   
    
    
    ''' Plot '''
    '''======'''
    if(plot == True):
        read_pkl_results(track_pkl_sname, plot=True)
    

    n_bytes = os.path.getsize(track_pkl_sname)
    print('Pickle File Size [MB]:\n\t', np.round(n_bytes*10**-6, 3))



    return track_pkl_sname
    
    




def simulate_PSI_with_crater(crater_profile_fname, v0s, degs, x0s=[0], y0s=[1], dt=0.01, seconds=10, n_free_cores=4, plotting_width=1.5, plot=True, old_tracks_img=False):
    ''' Simulate a plume surface interaction using a given crater profile
    and a list of initial starting parameters. All units in meters, seconds, and m/s
        
    Inputs:
        crater_profile_fname [str]   fullpath to the .pkl file with the crater profile
        v0s                  [array] array of initial velocity magnitudes
        degs                 [array] array of initial velocity angles (from horizontal)
        x0s                  [array] array of init x positions
        y0s                  [array] array of init y positions
        dt                   [float] time step [s] for ballistic trajectory
        seconds              [float] unused - number of seconds to run simulation
        n_free_cores         [int]   number of cores to leave free, worst case scenario, this function only uses one core
        plotting_width       [float] limit for plotting (-xlim,xlim) = (-crater.width * plotting_width, crater.width * plotting_width)
        plot                 [bool]  if True, plot results
        old_tracks_img       [bool]  if True, save the particle tracks image the onld way (normalized to 255), probably best to use False
    
    
    '''
        # Determine number of cores to use
    n_cores = os.cpu_count() - n_free_cores
    if(n_cores <= 0):
        n_cores = 1
    #print('Number of Cores To Use:\n\t', str(n_cores))
    
        # 
    short_crater_name = os.path.split(crater_profile_fname)[1]
    t0 = dtm.now() 
    
    ''' Open crater and plot '''
    '''======================'''
    with open(crater_profile_fname, 'rb') as inp:    
        crater_profile = pickle.load(inp)
    
	    # get crater info
    x      = crater_profile.x
    y      = crater_profile.y
    theta  = crater_profile.theta
    width  = crater_profile.width
    depth  = crater_profile.depth
    maxdeg = crater_profile.max_theta
    l_rim  = crater_profile.l_rim
    r_rim  = crater_profile.r_rim
    
        # Find safe landing Zone
    '''
    within_crater = np.where(abs(x) < width*0.38 )
    safe_list     = find_safe_landing(x[within_crater], theta[within_crater], max_slope=6) # [safe_total, start, end]
    safe_total    = abs(abs(safe_list[1]) - abs(safe_list[2])) 
    safe_center   = (safe_list[1] + safe_list[2]) /2

    if(safe_total >= 10):
        x -= safe_center
        crater_profile.x = x
    '''
    
    print('\n\n')
    print('===============')
    print('Target Crater: \n\t', short_crater_name)
    print('               \tCrater Diam:', crater_profile.width )
    print('===============')
    
    crater_profile_neg = copy.deepcopy(crater_profile)

    label_str = ('\nMax Slope: ' + str(np.round(maxdeg, 3)) + ' [$\degree$]' + 
               '\nDiameter: ' + str(width) + ' [m]' +
               '\nDepth: ' + str(np.round(depth, 1)) + ' [m]' +
               '\nd/D: ' + str(np.round(depth/width, 3)) +
               '\nTotal Incline: '  + str(np.round(np.rad2deg(np.arctan(2*depth/(width))), 3) ))
    		     
    rev_x     = np.asarray(list(reversed(x))) * -1
    rev_y     = np.asarray(list(reversed(y)))
    rev_theta = np.asarray(list(reversed(theta)))
    
    rev_crater_profile = Crater_Profile(rev_x, rev_y, theta, width, depth, 'rev', maxdeg, -1*r_rim, abs(l_rim))


    ''' Collect Particles and Prep saving '''
    '''==================================='''
    ret_list     = mp.Manager().list() 
    rev_ret_list = mp.Manager().list()
    particle_list     = []
    rev_particle_list = []

    for y0 in y0s:
        for x0 in x0s:
            for v0 in v0s:
    	        for deg in degs:
    		        vx0 = v0 * np.cos(np.deg2rad(deg))
    		        vy0 = v0 * np.sin(np.deg2rad(deg))
    		        particle_list.append(    [vx0, vy0, x0, y0, dt, seconds,     crater_profile,     ret_list, width*plotting_width])
    		        rev_particle_list.append([vx0, vy0, x0, y0, dt, seconds, rev_crater_profile, rev_ret_list, width*plotting_width])
    

    
    
    print('N particles:\n\t', len(particle_list)*2)

        # list directory structure
    crater_pkl_dir = os.path.split(crater_profile_fname)[0]
    profile_dir    = os.path.split(crater_pkl_dir)[0]
    
        # make track dir and file save names
    track_pkl_dir = os.path.join(profile_dir, 'track_plots_and_pkl')
    if(not os.path.exists(track_pkl_dir)):
        os.mkdir(track_pkl_dir)
    
    track_pkl_sname  = os.path.join(track_pkl_dir, ('tracks_' + short_crater_name))
    track_plot_sname = track_pkl_sname.replace('.pkl','.png')
    
        # Make hillshade
    '''
    hill_fname = os.path.join(track_pkl_dir, 'hillshade_' + ID_str) 
    hill_arr, ID_str = IMGLIB.grab_crater_hillshade(crater_profile_fname, pad_coeff=2)
    HILL_IMG = PIL.Image.fromarray(hill_arr)
    HILL_IMG.save(hill_fname)
    '''
    
        # Prep the lists for eventual saving
    hist_impact_x = []
    hist_impact_v = []

    save_init_vx0 = []
    save_init_vy0 = []
    save_impact_x = []
    save_impact_y = []
    save_impact_v = []
    save_impact_slope = []
    save_impact_theta = []
    
    shadow_bins_pos = np.arange(0, plotting_width*width, 1)
    shadow_bins_neg = np.arange(0, plotting_width*width, 1) * -1
    lowest_ejecta_pos = [[] for bins in shadow_bins_pos]
    lowest_ejecta_neg = [[] for bins in shadow_bins_neg]
    heat_img_pos = np.zeros((int(width*plotting_width), int(width*plotting_width)))
    heat_img_neg = np.zeros((int(width*plotting_width), int(width*plotting_width)))
    
    ''' Run Simulation and Collect info from tracks '''
    '''============================================='''
    
        #################################
        # Simulate particles to the right
    work_pool = mp.Pool(processes=n_cores)
    work_pool.map(run_particle_sim_array, particle_list)
    
    time.sleep(1)

        # Go through each track and plot
    k = 0
    for track in ret_list:
        k += 1
        
        if(k%50 == 0):
            print('\tCalculating Right Side...', str(k), ' / ', str(len(ret_list)), end='\r')
            # record other save params
        save_init_vx0.append(     track.vx0)
        save_init_vy0.append(     track.vy0)
        save_impact_x.append(     track.impact_x)
        #save_impact_y.append(     track.impact_y)
        save_impact_v.append(     track.impact_v)
        save_impact_slope.append( track.impact_slope)
        save_impact_theta.append( track.impact_theta)
                
        this_impact = track.impact_x
        
            # if the track impacted within the area of interest (crater_width * the plotting width)
            #  record the impact site for later histogram
            #  and note the lowest track hewight for an x bin
        if(len(track.x) > 1):
            hist_impact_x.append(track.impact_x)
            hist_impact_v.append(track.impact_v)    
    
                # record track on heatmap
            track_img = np.zeros((int(width*plotting_width), int(width*plotting_width) ), dtype=bool)
            for i in range(len(track.x)):
                if((int(track.y[i]) < width*plotting_width-1) and
                   (int(track.x[i]) < width*plotting_width-1)):
                    track_img[int(track.y[i] + (width*plotting_width*0.5)), int(track.x[i])] += 1
            
            heat_img_pos += track_img
   
                # Go through each of the shadow bins
            for j in range(len(shadow_bins_pos)):
                mask = np.where((track.x >= j) & (track.x <= j+1))
                if(len(mask[0]) >  0):
                    lowest_ejecta_pos[j].append(np.min(track.y[mask]))
            track = []
                    
        else:
            #print('No Return:', track) 
            continue

    
    ################################
        # Simulate particles to the left
    work_pool = mp.Pool(processes=n_cores)
    work_pool.map(run_particle_sim_array, rev_particle_list)

    time.sleep(1)
    
        #Go through each left track and plot
    k = 0
    for track in rev_ret_list:
        k += 1
        if(k%50 == 0):
            print('\t Calculating Left Side...', str(k), ' / ', str(len(rev_ret_list)), end='\r')
            # record other save params
        save_init_vx0.append(     -track.vx0)
        save_init_vy0.append(      track.vy0)
        save_impact_x.append(     -track.impact_x)
        #save_impact_y.append(      track.impact_y)
        save_impact_v.append(     -track.impact_v)
        save_impact_slope.append( -track.impact_slope)
        save_impact_theta.append( -track.impact_theta)
                
        this_impact = -track.impact_x
        
            # if the track impacted within the area of interest (crater_width * the plotting width)
            #  record the impact site for later histogram
            #  and note the lowest track hewight for an x bin
        if(len(track.x) > 1):
            hist_impact_x.append(-track.impact_x)
            hist_impact_v.append(-track.impact_v)    
    
                # record track on heatmap
            track_img = np.zeros((int(width*plotting_width), int(width*plotting_width) ), dtype=bool)
            for i in range(len(track.x)):
                if((int(track.y[i]) < width*plotting_width-1) and
                   (int(track.x[i]) < width*plotting_width-1)):
                    track_img[int(track.y[i] + (width*plotting_width*0.5)), int(track.x[i])] += 1
            
            heat_img_neg += track_img
            
                # Go through each of the shadow bins
            for j in range(len(shadow_bins_neg)):
                mask = np.where((track.x >= j) & (track.x <= j+1))
                if(len(mask[0]) >  0):
                    lowest_ejecta_neg[j].append(np.min(track.y[mask]))
            track = []
            
        else:
            #print('No Return:', track) 
            continue

    
    sim_t  = dtm.now()
    sim_dt = sim_t - t0
    print('Simulation Time:\n\t', sim_dt)
    
    
    ''' Compile Gap Info '''
    '''=================='''
        # Compile lowest ejecta        
    for j in range(len(lowest_ejecta_pos)):
        if(len(lowest_ejecta_pos[j]) == 0):
            lowest_ejecta_pos[j].append(width*plotting_width)
        lowest_ejecta_pos[j] = np.min(lowest_ejecta_pos[j])
    lowest_ejecta_pos = np.asarray(lowest_ejecta_pos)

    for j in range(len(lowest_ejecta_neg)):
        if(len(lowest_ejecta_neg[j]) == 0):
            lowest_ejecta_neg[j].append(width*plotting_width)
        lowest_ejecta_neg[j] = np.min(lowest_ejecta_neg[j])
    lowest_ejecta_neg = np.asarray(lowest_ejecta_neg)

    shadow_bins_total   = np.concat([list(reversed(shadow_bins_neg)),   shadow_bins_pos])
    lowest_ejecta_total = np.concat([list(reversed(lowest_ejecta_neg)), lowest_ejecta_pos])

    crater_floor  = np.asarray([calc_crater_wall(x + 0.5, crater_profile)[0] for x in shadow_bins_total])    
    gaps          = lowest_ejecta_total - crater_floor


    ''' Compile Heat Imgs '''
    '''==================='''
    if(old_tracks_img==True):
                # Pos
        heat_img_pos  = heat_img_pos / np.max(heat_img_pos)
        heat_img_pos *= 3/4
        heat_img_pos += 1/4
        heat_img_pos = np.where(heat_img_pos == np.min(heat_img_pos), 0, heat_img_pos)
    
        wht_img_pos  = np.ones((int(width*plotting_width), int(width*plotting_width), 3))
        wht_img_pos[:,:,0] -= heat_img_pos
        wht_img_pos[:,:,1] -= heat_img_pos
    
                # Neg
        heat_img_neg  = heat_img_neg / np.max(heat_img_neg) #########################################################
        heat_img_neg *= 3/4
        heat_img_neg += 1/4
        heat_img_neg = np.where(heat_img_neg == np.min(heat_img_neg), 0, heat_img_neg)
    
        wht_img_neg  = np.ones((int(width*plotting_width), int(width*plotting_width), 3))
        wht_img_neg[:,:,0] -= heat_img_neg
        wht_img_neg[:,:,1] -= heat_img_neg
    
            # Total
        IMG1 = PIL.Image.fromarray((wht_img_pos*255).astype(np.uint8))
        IMG2 = PIL.Image.fromarray((wht_img_neg*255).astype(np.uint8))
        IMG2 = IMG2.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
            
        final_IMG = PIL.Image.new('RGB', (wht_img_pos.shape[1]*2, wht_img_pos.shape[0]))
        final_IMG.paste(IMG2, (0,                    0))
        final_IMG.paste(IMG1, (wht_img_pos.shape[1], 0))
        
        final_img_arr = np.asarray(final_IMG) / 255.

    if(old_tracks_img != True):
        final_img_arr = np.zeros((heat_img_pos.shape[0], heat_img_pos.shape[1]*2))
        final_img_arr[:, 0:heat_img_pos.shape[1]]  = np.flip(heat_img_neg, axis=1) 
        final_img_arr[:,   heat_img_pos.shape[1]:] = heat_img_pos                 
        

        

    ''' Save tracks info '''
    '''=================='''
    file_obj = open(track_pkl_sname, 'wb')    

    short_mask   = np.where(abs(x) < width*plotting_width)
    short_crater = Crater_Profile(x[short_mask], y[short_mask], theta[short_mask], width, depth, crater_profile.ID_string, crater_profile.max_theta, l_rim, r_rim)
    
    swarm = Particle_Swarm(
    
                   save_init_vx0, 
                   save_init_vy0,
                   save_impact_x,
                   save_impact_y,
                   save_impact_slope,
                   save_impact_theta,
                   shadow_bins_total,
                   gaps,
                   final_img_arr,
                   save_impact_v)    
    
    pickle.dump(swarm,          file_obj)
    pickle.dump(short_crater, file_obj)
    
    file_obj.close()
   
    save_dt = dtm.now() - sim_t
    print('Save time:\n\t', save_dt)   
    
    
    ''' Plot '''
    '''======'''
    if(plot == True):
        read_pkl_results(track_pkl_sname, plot=True)
    

    n_bytes = os.path.getsize(track_pkl_sname)
    print('Pickle File Size [MB]:\n\t', np.round(n_bytes*10**-6, 3))



    return track_pkl_sname
    
    
#######################################################
#######################################################
#######################################################


def simulate_PSI_with_flat(flat_sname, v0s, degs, x0s=[0], y0s=[0.5], dt=0.01, seconds=10, n_free_cores=4, width=1000, plotting_width=1.5, plot=True):
    ''' Simulate PSI on flat surface. All units in meters, seconds, or m/s
    
    Inputs:
        flat_sname     [str]   fullpath to the .pkl file with the crater profile
        v0s            [array] array of initial velocity magnitudes
        degs           [array] array of initial velocity angles (from horizontal)
        x0s            [array] array of init x positions
        y0s            [array] array of init y positions
        dt             [float] time step [s] for ballistic trajectory
        seconds        [float] unused - number of seconds to run simulation
        n_free_cores   [int]   number of cores to leave free, worst case scenario, this function only uses one core
        width          [float] imaginary width of the 'crater' (helps with determining extent of plotting)
        plotting_width [float] limit for plotting (-xlim,xlim) = (-width * plotting_width, width * plotting_width)
        plot           [bool]  if True, plot results
         
    Outputs:
        track_pkl_sname [str] fullpath to .pkl file where the simulation results are saved
     
    '''
        # Create flat crater profile
    x         = np.arange(-100_000, 100_000, 10)
    y         = np.zeros(len(x))
    slope     = np.zeros(len(x)-1)
    width     = np.float64(width) 
    depth     = np.float64(0.)
    ID_string = 'flat'
    max_theta = np.float64(0.)
    l_rim     = width/-2
    r_rim     = width/ 2
    
    flat_profile = Crater_Profile(x, y, slope, width, depth, ID_string, max_theta, l_rim, r_rim)    
        
        # Save the flat profile
    file_obj = open(flat_sname, 'wb')    
    pickle.dump(flat_profile, file_obj)
    file_obj.close()
       
        # Run the simulation 
    track_pkl_sname = simulate_PSI_with_crater(flat_sname, v0s, degs, plotting_width=plotting_width)


    return track_pkl_sname


#######################################################
#######################################################
#######################################################


def simulate_PSI_with_wall(wall_params, v0s, degs, x0s=[0], y0s=[0.5], dt=0.01, seconds=10, n_free_cores=4, plotting_width=1.5, plot=True):
    '''Simulates a PSI with artificial walls. 
    
    Inputs:
        wall_params    [list]  contains [wall_height, wall_diam] in meters
        v0s            [list]  initial velocity magnitudes in m/s
        degs           [list]  initial velocity angles in  m/s
        x0s            [list]  starting x positions in m
        y0s            [list]  starting y positions in m
        dt             [float] NOT USED time step in s, now auto calculated
        seconds        [float] NOT USED ttotal time to run simulation, now auto calculated
        n_free_cores   [int]   number of cores to not use (leave free for other processes)
        plotting_width [float] plotting_xlim = plotting_width * crater_diam
        plot           [bool]  Y/N for generating plot
        
    Outputs:
    
        [file.pkl] file the has the particle tracks and result info
        [plot.png] plotted results (if plot==True)
    
    '''
    
    wall_height = wall_params[0]
    wall_diam   = wall_params[1]
    
        # make saving directories
    wall_profiles = 'wall_profiles'
    wall_plots    = os.path.join(wall_profiles, 'crater_plots_and_pkl')
    if(not os.path.exists(wall_profiles)):
        os.mkdir(wall_profiles)
    if(not os.path.exists(wall_plots)):
        os.mkdir(wall_plots)
    
    ID_string = ('Wall-H_' + str(wall_height) + '-D_' + str(wall_diam) + '.pkl')
    wall_pkl = os.path.join(wall_plots, ID_string)
    
        # create wall profile
    x = np.arange(-wall_diam*plotting_width-1, wall_diam*plotting_width+2, 1)
    x = np.concat(([-100000], x, [100000]))
    y = np.zeros(len(x))
    y[np.where(x==int(-wall_diam/2)   )] = wall_height
    y[np.where(x==int(-wall_diam/2) +1)] = wall_height
    y[np.where(x==int(-wall_diam/2) -1)] = wall_height
    y[np.where(x==int( wall_diam/2)   )] = wall_height
    y[np.where(x==int( wall_diam/2) +1)] = wall_height
    y[np.where(x==int( wall_diam/2) -1)] = wall_height
    
    slope = np.zeros(len(x)-1, dtype=float)
    
        # Save wall profile
    wall_profile = Crater_Profile(x, y, slope, float(wall_diam), float(wall_height), ID_string.replace('.pkl',''), max_theta=0.)    
    
    file_obj = open(wall_pkl, 'wb')    
    pickle.dump(wall_profile, file_obj)
    file_obj.close()
    
    simulate_PSI_with_crater(wall_pkl, v0s, degs, x0s=x0s, y0s=y0s, dt=dt, seconds=seconds, 
                             n_free_cores=n_free_cores, plotting_width=plotting_width, plot=plot)


    return


#######################################################
#######################################################
#######################################################



def simulate_PSIs_on_DTM(DTM_dir, v0s, degs, overwrite=False, max_crater_diam=4000, cool_time=60, max_deg=20):
    ''' Simulates PSIs on craters identified for a DTM image, saves .pkl files and plots 
    
    Inputs:
        DTM_dir         [str]   fullpath to directory containing DTM and labels
        v0s             [list]  initial speeds in m/s
        degs            [list]  degrees above horizontal for initial velocities
        overwrite       [bool]  Y/N to overwirte previous results
        max_crater_diam [int]   maximum crater diameter to run simulation for (in meters)
        max_deg         [float] maximum slope for crater [deg]
        cool_time       [int]   time in between crater simulations to rest (so you won't nuke your CPU)
        
        
        
    Outputs:
        creates subdirs in DTM_dir that have the pkl files with track in fo and simulation results, also the plots

    '''
    profile_dir = os.path.join(DTM_dir, os.path.split(DTM_dir)[1].replace('a_','') + '_profiles')
    
    crater_dir = os.path.join(profile_dir, 'crater_plots_and_pkl')
    track_dir  = os.path.join(profile_dir, 'track_plots_and_pkl')
        
        # Check if profiles exist, exit if they don't
    if(not os.path.exists(crater_dir)):
        print('Please create crater profiles.... Exiting')
        return
    
    profile_fnames = list_files(crater_dir , '.pkl')
    
    sorted_fname = os.path.join(profile_dir, 'sorted_results.csv')
    sorted_df= []
    if(os.path.exists(sorted_fname)):
        sorted_df = pd.read_csv(sorted_fname)    
    
    
    for fname in profile_fnames:
        track_fname = os.path.join(track_dir, 'tracks_' + os.path.split(fname)[1])
        with open(fname, 'rb') as inp:    
            crater_profile = pickle.load(inp)
        
            # Check for acceptable crater slope    
        r_side = crater_profile.theta[np.where((crater_profile.x >= 0) & (crater_profile.x <= crater_profile.r_rim) )]
        l_side = crater_profile.theta[np.where((crater_profile.x <= 0) & (crater_profile.x >= crater_profile.l_rim) )]

        pos_deg = np.max([abs(r_elm) for r_elm in r_side])
        neg_deg = np.max([abs(l_elm) for l_elm in l_side])


            
        if(len(sorted_df) != 0):
            current_line = sorted_df[sorted_df['pkl_fname'] == os.path.split(track_fname)[1]]
            
            if(len(current_line['status']) != 0):
                if(current_line['status'].iloc[0] == 'N'):
                    print(os.path.split(fname)[1], ' has already been deemed unworthy, skipping....')
                    continue


        if(pos_deg > max_deg and
           neg_deg > max_deg):
            print('Crater is too steep on both sides, skipping....')
            continue
            
            # Crater Size check
        if(crater_profile.width > max_crater_diam):
            print(os.path.split(fname)[1], ' is too large, skipping...')
            continue
        
            # File exists, but don't overwrite 
        if(os.path.exists(track_fname) and
           overwrite != True):
            print('Skipping already complete: ', os.path.split(track_fname)[1])
            continue
             
            # File exsits, and overwrite
        if(os.path.exists(track_fname) and
           overwrite == True):
            print('Overwriting: ', os.path.split(track_fname)[1])
            simulate_PSI_with_crater(fname, v0s, degs)
            continue
            
            # File doesn't exist simulate PSI
        if(not os.path.exists(track_fname)):
            print('Starting New Simulation: ', os.path.split(fname)[1])               
            simulate_PSI_with_crater(fname, v0s, degs)
    
        time.sleep(cool_time)
    
    return


#######################################################
#######################################################
#######################################################


def read_pkl_results(pkl_fname, plot=False, save_track_img=True, custom_title=False):
    ''' Reads the pkl results for a simulation on a crater
    
    Inputs:    
        pkl_fname      [str]  fullpath to the .pkl file with saved simulation results
        plot           [bool] Y/N for saving a plot showing the results
        save_track_img [bool] Y/N for saving a plot showing the results
        custom_title   [str]  title for the entire plot
        
    Outputs:
        [plot.png]
        results        [obj] Particle_Swarm Object with all the reults from the simulation
        crater_profile [obj] Crater_Profile Object with crater info
        
    '''
    
        # Open and copy results
    with open(pkl_fname, 'rb') as inp:    
        results         = pickle.load(inp)
        crater_profile  = pickle.load(inp)

    init_vx      = results.init_vx
    init_vy      = results.init_vy
    impact_v     = results.impact_v
    impact_x     = results.impact_x
    impact_y     = results.impact_y
    impact_slope = results.impact_slope
    impact_angle = results.impact_angle
    shadow_x     = results.shadow_x
    shadow_y     = results.shadow_y
    heat_img     = results.heat_img
 
    max_deg = crater_profile.max_theta
    width   = crater_profile.width
    depth   = crater_profile.depth
    slope   = crater_profile.theta   
    l_rim   = crater_profile.l_rim
    r_rim   = crater_profile.r_rim
    
    impact_x  = np.asarray(impact_x)
        
    xlim = int(heat_img.shape[1]/2)

    zero_mask = np.where(heat_img==0)
    disp_img = np.log10(heat_img+1)
    disp_img  = (disp_img / np.max(disp_img) ) #* 255 #+ 255
    disp_img[zero_mask] = 0
    

    
    '''
        # open hillshade image of crater
    profile_dir       = os.path.join(os.path.split(os.path.split(pkl_fname)[0])[0], 'crater_plots_and_pkl')
    profile_pkl_fname = os.path.join(profile_dir, os.path.split(pkl_fname)[1].replace('tracks_', ''))
    if('flat' not in profile_pkl_fname and
       'Wall' not in profile_pkl_fname):
        img_arr, ID_str   = IMGLIB.grab_crater_hillshade(profile_pkl_fname,)
    '''

    if(plot == True):
        dD = depth/width
        if(width == 0):
            dD = 0 
        label_str = ('depth: ' + str(np.round(depth,1)) + ' [m]' +
                     '\nDiameter: ' + str(width) + ' [m]' + 
                     '\nd/D: ' + str(np.round(depth/width,3)) 
                     )
        
        crater_mask = np.where(abs(np.asarray(results.shadow_x)) < crater_profile.width*0.4)
        in_crater_x = results.shadow_x[crater_mask]
        theta = np.asarray([calc_crater_wall(this_x, crater_profile)[1] for this_x in in_crater_x])
 
            # Ideal landing space    (>  6 deg)
        safe_ideal =   find_safe_landing( in_crater_x, theta, 6)
            # Shadow 1m tall rover
        safe_1m  = find_safe_shadow(results.shadow_x,  results.shadow_y, 1)
            # Shadow 5m ISS module
        safe_5m  = find_safe_shadow(results.shadow_x,  results.shadow_y, 5)
            # Shadow 8m Blue Moon Mk1
        safe_8m  = find_safe_shadow(results.shadow_x,  results.shadow_y, 8)
            # Shadow 16m Blue Moon Mk2
        safe_16m = find_safe_shadow(results.shadow_x, results.shadow_y, 16)
        
            # Get average impact velocity every 10 m
        bins_impact_vs = np.arange(-xlim,xlim+10,10)
        
        avg_impact_vs = np.zeros(len(bins_impact_vs)-1)
        std_impact_vs = np.zeros(len(bins_impact_vs)-1)
        pos_impact_vs = np.zeros(len(bins_impact_vs)-1)

        impact_v = np.asarray(impact_v)


        for i in range(len(avg_impact_vs)):
            pos_impact_vs[i] = np.mean(bins_impact_vs[i:i+2])
            
            impact_bin = impact_v[  np.where((impact_x > bins_impact_vs[i]) &  
                                             (impact_x < bins_impact_vs[i+1])) ]
            if(len(impact_bin) != 0):
                avg_impact_vs[i] = abs(np.mean(impact_bin))         
                std_impact_vs[i] = np.std(impact_bin)
            else:
                avg_impact_vs[i] = 0
                std_impact_vs[i] = 0
        
        crater_radius  = crater_profile.width/2
        #major_ticks = np.array([crater_profile.l_rim, crater_profile.r_rim])
        major_ticks = np.concatenate( [np.arange(-(crater_radius)*10, -crater_radius, crater_radius),
                                       [crater_profile.l_rim],
                                       [0],
                                       [crater_profile.r_rim],
                                       np.arange(crater_radius*2, crater_radius*10, crater_radius)
                                        ])     
        
        
        
            # Plot
        plt.clf()
        fig, axs = plt.subplots(4, sharex=True, figsize=(8,10))
                # Track plot (top)
                ###################
        i_track = 0
        axs[i_track].set_xticks(major_ticks)
        axs[i_track].set_yticks(np.arange(-1000, 1000, 50))
        axs[i_track].grid()
        axs[i_track].plot(crater_profile.x, crater_profile.y, 'k-', label=label_str)
        axs[i_track].plot(l_rim, calc_crater_wall(l_rim, crater_profile)[0], color=colors[1], marker='x')
        axs[i_track].plot(r_rim, calc_crater_wall(r_rim, crater_profile)[0], color=colors[1], marker='x')
        axs[i_track].imshow(disp_img, origin='lower', cmap='Blues', 
                                        extent=(-int(disp_img.shape[1]/2), int(disp_img.shape[1]/2), 
                                                -int(disp_img.shape[0]/2), int(disp_img.shape[0]/2)))    # extent=[left, right, bottom, top]
        '''
        axs[i_track].fill_between(crater_profile.x, 
                                  crater_profile.y, 
                                  np.ones(len(crater_profile.y))*-1000, color='grey')
        '''
            

        axs[i_track].set_title('Particle Heatmap')
        axs[i_track].set_ylabel('Elevation [m]')
        axs[i_track].axis('equal')
        axs[i_track].set(xlim=(-xlim, xlim), )# ylim=(-10, depth*3))
        axs[i_track].set_ylim(bottom = -int(width/2))
        axs[i_track].legend(loc='lower left',)
        axs[i_track].set_xlabel('Distance [m]')

        
                # impact position hist plot (middle)
                #  and impact velocties
                ####################################
        i_hist = 2
        axs[i_hist].set_xticks(major_ticks)
        axs[i_hist].set_yticks(np.arange(-1000, 1000, 50))
        axs[i_hist].grid()
        axs[i_hist].set_title('Particle Impact') 
        axs[i_hist].set_ylabel('Number of Impacts & \n Impact Velocities [m/s]')
        axs[i_hist].set_yscale('log')    
        bin_vals = axs[i_hist].hist(impact_x[np.where(abs(impact_x) < xlim)], bins=np.arange(-xlim, xlim+1),  )#color=colors[2]) 
        axs[i_hist].errorbar(pos_impact_vs, avg_impact_vs, 3*std_impact_vs, fmt='bo')
        axs[i_hist].set_xlabel('Distance [m]')
        axs[i_hist].set(ylim=(0,10*np.max((np.max(avg_impact_vs), np.max(bin_vals[0])))))
    

    
                # Ejecta shadow
                ###############
        i_shadow = 3
        axs[i_shadow].set_xticks(major_ticks)
        axs[i_shadow].set_yticks(np.arange(-1000, 1000, 50))
        axs[i_shadow].grid()
        axs[i_shadow].set_title('Umbrella Zones')
        axs[i_shadow].vlines(shadow_x, np.zeros(len(shadow_x)), shadow_y)
        axs[i_shadow].set_ylabel('Height [m]')
        axs[i_shadow].set(xlim=(-xlim, xlim), ylim=(0, 16))
        axs[i_shadow].set_xlabel('Distance [m]')
        axs[i_shadow].set_yticks((1, 5, 8, 16))
        

                    # Go through each largest umbrella zone for each height and annotate with text 
        for umbrella, meter in zip([safe_1m, safe_5m, safe_8m, safe_16m], [1,5,8,16]):
            if(umbrella[0] > 0):
                umb_x = np.arange(umbrella[1], umbrella[2], 1)
                umb_y = np.ones(len(umb_x)) * meter 
                axs[i_shadow].plot(umb_x, umb_y, 'k-')
                #axs[i_shadow].vlines(umbrella[1], 0, meter)
                #axs[i_shadow].vlines(umbrella[2], 0, meter)
                if(umbrella[1] > 0):
                    alignment = 'right'
                if(umbrella[1] <=0):
                    alignment = 'left'
                axs[i_shadow].text(umbrella[1], meter , 'Zone ' +str(meter) + ' X ' + str(umbrella[0]) + ' m ' ,  
                         ha=alignment, style='italic')
    
    
    
                # Crater Slope
                ###############
        i_slope = 1

        slope_lbl = ('\nMax Slope: ' + str(np.round(max_deg,3)) + '$\degree$' +
                     '\nTotal Incline: '  + str(np.round(np.rad2deg(np.arctan(2*depth/(width))), 3) ) + '$\degree$')
        axs[i_slope].set_xticks(major_ticks)
        axs[i_slope].set_yticks(np.array([0, 6, 20]))
        axs[i_slope].grid()

        axs[i_slope].set_title('Crater Slope')
        axs[i_slope].scatter(crater_profile.x, abs(slope), marker='o',      color='black', 
                                                  label='Local Slope')#label=slope_lbl)
        axs[i_slope].plot([-xlim, xlim], np.ones(2)*20, linestyle='--',  color=colors[2], 
                                                  label='Maximum LTV Slope [20$\degree$]')
        axs[i_slope].plot([-xlim, xlim], np.ones(2)*6,  linestyle='-.',  color=colors[6], 
                                                  label='Ideal Landing Limit [6$\degree$]')
        axs[i_slope].text(0, -6, str(safe_ideal[0]) + ' m Landing Zone', style='italic', ha='center')
        
        axs[i_slope].set_ylabel('Slope [$\degree$]')
        axs[i_slope].set_xlabel('Distance [m]')
        axs[i_slope].set(xlim=(-xlim, xlim), ylim=(-10, 50))
        axs[i_slope].legend(loc='upper left')

        
        '''
        i_img = 4
        axs[i_img].imshow(img_arr, origin='lower', extent=(-int(img_arr.shape[1]/2), int(img_arr.shape[1]/2), 
                                                           -int(img_arr.shape[0]/2), int(img_arr.shape[0]/2)) )
        '''
        
        
        fig_title_str = os.path.split(pkl_fname)[1].replace('.pkl','').replace('tracks_','') 
        if(custom_title != False):
            fig_title_str = custom_title
        fig.suptitle(fig_title_str)
        plt.tight_layout()
        
        plt.savefig(pkl_fname.replace('.pkl','.png'))
        plt.show()
        plt.close(fig)
    
    return results, crater_profile


#######################################################
#######################################################
#######################################################


def find_safe_shadow(shadow_x, shadow_y, height):
    ''' Finds an area where particles do not pass through (an umbrella zone)
        
    Inputs:
        shadow_x [array] x positions for crater profile
        shadow_y [array] gap between crater height and lowest ejecta for each x position in shadow_x
        height   [float] minimum height for safe umbrella zone
        
    Outputs: (Contained in a list)
        safe_total [int] distance for largest area that satifies safe height requirement
        safe_start [int] start position of safe zone
        safe_end   [int] end position fo safe zone
        
    '''
    
    
    safe_total = 0
    safe_start = 0
    safe_end   = 0
    mask = np.where(shadow_y > height)[0]
    
    if(len(mask) > 0):
        split_mask = np.split(mask, np.where(np.diff(mask) != 1)[0]+1 )   
    
        for this_split in split_mask:
            if(len(this_split) > safe_total):
                safe_total = len(this_split)
                safe_start = this_split[0]
                safe_end   = this_split[-1]
    
    return [safe_total, shadow_x[safe_start], shadow_x[safe_end]]


#######################################################
#######################################################
#######################################################


def find_safe_landing(shadow_x, theta, max_slope):
    ''' Finds a safe area to land within a crater
    
    Inputs:
        shadow_x  [array] x positions for the crater profile
        theta     [array] instantaneous slope for the crater (each element is a float)
        max_slope [flaot] limit for acceptable slope
        
    Outputs: (Contained in a list)
        safe_total [int] distance for the largest section of a safe landing zone
        safe_start [int] position where the safe area starts
        safe_end   [int] position where the safe area ends
    
    '''
    
    safe_total = 0
    safe_start = 0
    safe_end   = 0
    
    mask = np.where(abs(theta) < max_slope)[0]
    
    if(len(mask) > 0):
        split_mask = np.split(mask, np.where(np.diff(mask) != 1)[0]+1 )
    
        for this_split in split_mask:
            if(len(this_split) > safe_total):
                safe_total = len(this_split)
                safe_start = this_split[0]
                safe_end   = this_split[-1]
        
    return [safe_total, shadow_x[safe_start], shadow_x[safe_end]]



#######################################################
#######################################################
#######################################################


def compare_results(DTM_dirs, lander_precision=50, gather_new=True, save_dir='total_results'):
    ''' Compiles and compares results from all simulations
    
    Inputs:
        DTM_dirs         [list] fullpaths to directories that contain a NAC DTM, and all the reesults for the simulations
        lander_precision [int]  NOT USED
        gather_new       [bool] Y/N if yes, seek new results for plotting ana analysis, otherwise used previous saved compliation
        save_dir         [str]  fullpath to directory where full results aare saved
        
    Outputs:
        [plot] .png plots that show the full results
    
    '''
    
    print('Comparing Results')
    plt.clf()
    
    try:
        summarize_labels(NAC_dir='NAC_DTM')
    except:
        print('Could not summarize labels')
    
    compare_results_fname = os.path.join(save_dir, 'compare_results.csv')
    impact_vel_fname      = os.path.join(save_dir, 'impact_velocities.csv')
    flat_pkl              = 'flat_profile/flat_profile_profiles/track_plots_and_pkl/tracks_flat_profile.pkl'
   
    
    all_craters_dir = os.path.join(save_dir, 'all_craters')
    safe_01m_craters_dir = os.path.join(save_dir, 'safe_01m_craters')
    safe_05m_craters_dir = os.path.join(save_dir, 'safe_05m_craters')
    safe_08m_craters_dir = os.path.join(save_dir, 'safe_08m_craters')
    safe_16m_craters_dir = os.path.join(save_dir, 'safe_16m_craters')
    
    
    if(not os.path.exists(all_craters_dir)):
        os.mkdir(all_craters_dir)
    if(not os.path.exists(safe_01m_craters_dir)):
        os.mkdir(safe_01m_craters_dir)
    if(not os.path.exists(safe_05m_craters_dir)):
        os.mkdir(safe_05m_craters_dir)
    if(not os.path.exists(safe_08m_craters_dir)):
        os.mkdir(safe_08m_craters_dir)
    if(not os.path.exists(safe_16m_craters_dir)):
        os.mkdir(safe_16m_craters_dir)


    if(not os.path.exists(save_dir)):
        os.mkdir(save_dir)
        
    if(gather_new == True):
        if(os.path.exists(save_dir)):
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
            os.mkdir(all_craters_dir)
            os.mkdir(safe_01m_craters_dir)
            os.mkdir(safe_05m_craters_dir)
            os.mkdir(safe_08m_craters_dir)
            os.mkdir(safe_16m_craters_dir)
                
        init_vx       = []
        init_vy       = []
        impact_x      = []
        impact_y      = []
        impact_slope  = []
        impact_angle  = []
        norm_impact_x = []
        shadow_x      = []
        shadow_y      = []
        #heat_img     = []
        impact_v      = []
        
        crater_name  = []
        crater_x     = []
        crater_y     = []
        crater_theta = []
        crater_width = []
        crater_depth = []
    
        safe_ideal    = []
        safe_possible = []
        safe_1m       = []
        safe_5m       = []
        safe_8m       = []
        safe_16m      = []
    
    
            # Move through each pkl dir (NAC IMG), and each crater result file
        for this_dir in DTM_dirs:
            profile_dir  = os.path.join(this_dir, os.path.split(this_dir)[1].replace('a_', '') + '_profiles')
            track_dir    = os.path.join(profile_dir, 'track_plots_and_pkl')
            result_fname = os.path.join(profile_dir, 'sorted_results.csv')
            
            if(not os.path.exists(result_fname)):
                print('No Sorting file found for: ', this_dir)
                continue
            print(os.path.split(this_dir))
            
            df = pd.read_csv(result_fname)
            
            
            pkl_fnames = df['pkl_fname'].loc[df['status'] == 'Y']
            pkl_fnames = [os.path.join(track_dir, pkl_fname) for pkl_fname in pkl_fnames]
            
                # Move through each crater
            for this_pkl in pkl_fnames:
                print('\t', os.path.split(this_pkl)[1])
                
                    # Read results and make a copy
                results, crater_profile = read_pkl_results(this_pkl)
                
                crater_name.append(os.path.split(this_pkl)[1])
                shadow_x.append(results.shadow_x)
                shadow_y.append(results.shadow_y)            
                impact_x.append(results.impact_x)
                impact_v.append(results.impact_v)
                impact_slope.append(results.impact_slope)
                impact_angle.append(results.impact_angle)
                crater_x.append(crater_profile.x)
                crater_y.append(crater_profile.y)
                crater_width.append(crater_profile.width)
                crater_depth.append(crater_profile.depth)
                crater_theta.append(crater_profile.theta)
                
                norm_impact_x.append(list( np.asarray(results.impact_x).flatten() / (crater_profile.width *0.5) ) )
                #shadow_x = shadow_x[0]
                #print(shadow_x[0])
                #print(results.shadow_x)
                
                crater_mask = np.where(abs(np.asarray(results.shadow_x)) < crater_profile.width*0.4)
                in_crater_x = results.shadow_x[crater_mask]
                theta = np.asarray([calc_crater_wall(this_x, crater_profile)[1] for this_x in in_crater_x])
     
                    # Ideal landing space    (>  6 deg)
                safe_ideal.append(   find_safe_landing( in_crater_x, theta, 6))
                    # Possible landing space (> 15 deg)
                safe_possible.append(find_safe_landing( in_crater_x, theta, 15))
            
                    # Shadow 1m tall rover
                safe_1m.append(find_safe_shadow(results.shadow_x,  results.shadow_y, 1))
                    # Shadow 5m ISS module
                safe_5m.append(find_safe_shadow(results.shadow_x,  results.shadow_y, 5))
                    # Shadow 8m Blue Moon Mk1
                safe_8m.append(find_safe_shadow(results.shadow_x,  results.shadow_y, 8))
                    # Shadow 16m Blue Moon Mk2
                safe_16m.append(find_safe_shadow(results.shadow_x, results.shadow_y, 16))


                
                    # copy to all_craters_dir
                short_pkl_fname = os.path.split(this_pkl)[1]
                
                dest = os.path.join(all_craters_dir, short_pkl_fname).replace('.pkl','.png')
                src  = this_pkl.replace('.pkl','.png')
                shutil.copyfile(src, dest)
                    
                    # Copy to specific umbrella crater dir
                if(safe_1m[-1][0] > 0):
                    dest = os.path.join(safe_01m_craters_dir, short_pkl_fname).replace('.pkl','.png')
                    shutil.copyfile(src, dest)
                    if(safe_5m[-1][0] > 0):
                        dest = os.path.join(safe_05m_craters_dir, short_pkl_fname).replace('.pkl','.png')
                        shutil.copyfile(src, dest)
                        if(safe_8m[-1][0] > 0):
                            dest = os.path.join(safe_08m_craters_dir, short_pkl_fname).replace('.pkl','.png')
                            shutil.copyfile(src, dest)
                            if(safe_16m[-1][0] > 0):
                                dest = os.path.join(safe_16m_craters_dir, short_pkl_fname).replace('.pkl','.png')
                                shutil.copyfile(src, dest)
                    
                
                
    
        impact_slope  = np.asarray(impact_slope).flatten()
        impact_angle  = np.asarray(impact_angle).flatten()
        impact_v      = np.asarray(impact_v).flatten()
        norm_impact_x = np.asarray(norm_impact_x).flatten()
    
        crater_width = np.asarray(crater_width).flatten()
        crater_depth = np.asarray(crater_depth).flatten()
    
        dD = crater_depth / crater_width
    
        safe_ideal    = np.asarray(safe_ideal)
        safe_possible = np.asarray(safe_possible)
        safe_1m       = np.asarray(safe_1m)
        safe_5m       = np.asarray(safe_5m)
        safe_8m       = np.asarray(safe_8m)
        safe_16m      = np.asarray(safe_16m)

            # Save results comparison
        df_results = pd.DataFrame()
        df_results['crater_name']   = crater_name
        df_results['crater_width']  = crater_width
        df_results['crater_depth']  = crater_depth
        df_results['dD']            = dD
        df_results['safe_ideal']    = safe_ideal[:,0];     df_results['safe_ideal_start']    = safe_ideal[:,1];      df_results['safe_ideal_end']    = safe_ideal[:,2] 
        df_results['safe_possible'] = safe_possible[:,0];  df_results['safe_possible_start'] = safe_possible[:,1];   df_results['safe_possible_end'] = safe_possible[:,2]
        df_results['safe_1m']       = safe_1m[:,0];        df_results['safe_1m_start']       = safe_1m[:,1];         df_results['safe_1m_end']       = safe_1m[:,2]
        df_results['safe_5m']       = safe_5m[:,0];        df_results['safe_5m_start']       = safe_5m[:,1];         df_results['safe_5m_end']       = safe_5m[:,2]
        df_results['safe_8m']       = safe_8m[:,0];        df_results['safe_8m_start']       = safe_8m[:,1];         df_results['safe_8m_end']       = safe_8m[:,2]
        df_results['safe_16m']      = safe_16m[:,0];       df_results['safe_16m_start']      = safe_16m[:,1];        df_results['safe_16m_end']      = safe_16m[:,2]
        df_results.to_csv(compare_results_fname, index=False)
        
            # Save impact velocities
        df_impact_v = pd.DataFrame()
        df_impact_v['impact_slope']  = impact_slope
        df_impact_v['impact_angle']  = impact_angle
        df_impact_v['impact_v']      = impact_v
        df_impact_v['norm_impact_x'] = norm_impact_x
        df_impact_v.to_csv(impact_vel_fname, index=False)

        


        # Open results comparison
    df_results  = pd.read_csv(compare_results_fname)
    df_impact_v = pd.read_csv(impact_vel_fname)

    max_zone    = np.max(df_results[['safe_1m', 'safe_5m', 'safe_8m', 'safe_16m']]) * 1.1
    max_landing = np.max(df_results[['safe_ideal', 'safe_possible']]) *1.1

        # Open results of flat run
    flat_results, flat_profile = read_pkl_results(flat_pkl)
    flat_impact_x = np.asarray(flat_results.impact_x) / (flat_profile.width /2)
    
        # print the min and max impact angles, slopes, and speeds
    print(np.min(np.absolute(df_impact_v['impact_slope'])), np.max(df_impact_v['impact_slope']  ))
    print(np.min(np.absolute(df_impact_v['impact_angle'])), np.max(df_impact_v['impact_angle']  ))
    print(np.min(np.absolute(df_impact_v['impact_v'])),     np.max(df_impact_v['impact_v']      ))
 
    print(np.max(df_results['dD']))
    

        # Plot Crater Attributes
        # x diam
        # y depth
    '''
    if(True):
        plt.clf()
            # no shadow zone            
        plt.scatter(df_results['crater_width'].loc[df_results['safe_1m']  == 0], 
                    df_results['crater_depth'].loc[df_results['safe_1m']  == 0],
                    color=colors[0],
                    marker=markers[0],
                    label='No Shadow')

            # 1 m shadow zone
        df_1m = df_results.loc[df_results['safe_1m'] >  0]
        df_1m = df_1m.loc[df_1m['safe_5m'] == 0]
        plt.scatter(#df_results['crater_width'].loc[df_results['safe_1m']  > 0], 
                    #df_results['crater_depth'].loc[df_results['safe_1m']  > 0],
                    df_1m['crater_width'],
                    df_1m['crater_depth'],
                    color=colors[1],
                    marker=markers[1],
                    label='1 m Shadow')
            # 5 m shadow zone
        df_5m = df_results.loc[df_results['safe_5m'] >  0]
        df_5m = df_5m.loc[df_5m['safe_8m'] == 0]
        plt.scatter(#df_results['crater_width'].loc[df_results['safe_5m']  > 0], 
                    #df_results['crater_depth'].loc[df_results['safe_5m']  > 0],
                    df_5m['crater_width'],
                    df_5m['crater_depth'],
                    color=colors[2],
                    marker=markers[2],
                    label='5 m Shadow')
            # 8 m shadow zone
        df_8m = df_results.loc[df_results['safe_8m'] >  0]
        df_8m = df_8m.loc[df_8m['safe_16m'] == 0]
        plt.scatter(#df_results['crater_width'].loc[df_results['safe_8m']  > 0], 
                    #df_results['crater_depth'].loc[df_results['safe_8m']  > 0],
                    df_8m['crater_width'],
                    df_8m['crater_depth'],
                    color=colors[3],
                    marker=markers[3],
                    label='8 m Shadow')
            # 16 m shadow zone
        plt.scatter(df_results['crater_width'].loc[df_results['safe_16m'] > 0], 
                    df_results['crater_depth'].loc[df_results['safe_16m'] > 0],
                    color=colors[4],
                    marker=markers[4],
                    label='16 m Shadow')
            # 0.1
        plt.plot(np.arange(0,3000), 0.1*np.arange(0,3000),   color='0.0', linestyle='-',   )#label='d/D')
        plt.text(1400, 125, '0.1', color='0.0', rotation=20)
            # 0.105
        plt.plot(np.arange(0,3000), 0.105*np.arange(0,3000), color='0.2', linestyle='-',  )#label='d/D: 0.105')
        plt.text(1200, 135, '0.105', color='0.2', rotation=25)
            # 0.12
        plt.plot(np.arange(0,3000), 0.12*np.arange(0,3000),  color='0.4', linestyle='-',   label='d/D')
        plt.text(1000, 130, '0.12', color='0.4', rotation=35)
            # 0.14
        plt.plot(np.arange(0,3000), 0.14*np.arange(0,3000),  color='0.6', linestyle='-',  )#label='d/D: 0.14')        
        plt.text(900, 140, '0.14', color='0.6', rotation=40)
            # 0.18
        plt.plot(np.arange(0,3000), 0.18*np.arange(0,3000),  color='0.8', linestyle='-',   )#label='d/D: 0.18')
        plt.text(800, 160, '0.18', color='0.8', rotation=45)

        plt.xlabel('Crater Diameter [m]')
        plt.ylabel('Crater Depth [m]')
        plt.title('Target Crater Sizes')
        plt.legend()
        plt.xlim((0, np.max(df_results['crater_width'])))
        plt.ylim((0, np.max(df_results['crater_depth'])))
        
        plt.savefig(os.path.join(save_dir, 'Target_Crater_Sizes.png'))
        plt.show()
        plt.clf()        
    '''



        # Plot Safe Zones (Shadow Zones) as a function of d/D with landing zone shading
    if(True):
        
        plt.clf()
        fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, figsize=(9,9))
        
        maxdD = 0.3
        
        axs1.set_title('Rover (1 m)')
        scatter1 = axs1.scatter(df_results['safe_1m'], df_results['dD'], s=40, c=df_results['safe_ideal'], cmap='Reds', edgecolors='Black')
        legend   = axs1.legend(*scatter1.legend_elements(), loc='upper right', title='Ideal Landing\n Zone Size [m]')
        #axs1.set(xlabel='Zone Size [m]')
        axs1.set(ylabel='depth / Diameter')
        axs1.set(xlim=(-10, max_zone))
        axs1.set(ylim=(0,maxdD))
        
        
        axs2.set_title('ISS-sized module (5 m)')
        scatter2 = axs2.scatter(df_results['safe_5m'], df_results['dD'], s=40, c=df_results['safe_ideal'], cmap='Greens', edgecolors='Black')
        legend   = axs2.legend(*scatter2.legend_elements(), loc='upper right', title='Ideal Landing\n Zone Size [m]')
        #axs2.set(xlabel='Zone Size [m]')
        #axs2.set(ylabel='depth / Diameter')
        axs2.set(xlim=(-10, max_zone))
        axs2.set(ylim=(0,maxdD))
    
        
        axs3.set_title('Blue Moon Mk1 (8m)')
        scatter3 = axs3.scatter(df_results['safe_8m'], df_results['dD'], s=40, c=df_results['safe_ideal'], cmap='Blues', edgecolors='Black')
        legend   = axs3.legend(*scatter3.legend_elements(), loc='upper right', title='Ideal Landing\n Zone Size [m]')
        axs3.set(xlabel='Zone Size [m]')
        axs3.set(ylabel='depth / Diameter')
        axs3.set(xlim=(-10, max_zone))
        axs3.set(ylim=(0,maxdD))
    
        
        axs4.set_title('Blue Moon Mk2 (16 m)')
        scatter4 = axs4.scatter(df_results['safe_16m'], df_results['dD'], s=40, c=df_results['safe_ideal'], cmap='Grays', edgecolors='Black')
        legend   = axs4.legend(*scatter4.legend_elements(), loc='upper right', title='Ideal Landing\n Zone Size [m]')
        axs4.set(xlabel='Zone Size [m]')
        #axs4.set(ylabel='depth / Diameter')
        axs4.set(xlim=(-10, max_zone))
        axs4.set(ylim=(0,maxdD))
    
    
        plt.suptitle('Ejecta Umbrella Zone Size By Height', size=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'Shadow_zones.png'))
        plt.show()
        
        
        
        # Plot ideal/possible landing zones
    if(True):
        plt.clf()
        
        fig, axs = plt.subplots(2, figsize=(6,6))
        
        fig.supylabel('Crater d/D', size=16)
        fig.supxlabel('Landing Zone Size [m]')
        fig.suptitle('Safe Landing Zones', size=16)
        
        axs[0].plot(df_results['safe_ideal'],    df_results['dD'], 'ro')
        axs[0].set_title('Ideal (Slope < 6$\degree$)')
        axs[0].set(xlim=(-10, max_landing))
        
        
        axs[1].plot(df_results['safe_possible'], df_results['dD'], 'bo')
        axs[1].set_title('Possible (Slope < 15$\degree$)')
        axs[1].set(xlim=(-10, max_landing))
        
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'Landing_zones.png'))
        plt.show()
        
        
        
        
        # Plot Histogram of impact locations normalized to 
    if(True):
        plt.clf()
            
            # Take the flat impact x locations and duplicate them so its as if I ran an equal number of 
            #  flat simulations as I did crater simulations
        n_sims = len(df_results)
        flat_impact_x = flat_impact_x * np.ones((n_sims, len(flat_impact_x)))

        #flat_impact_x = flat_impact_x[np.where(abs(flat_impact_x) <2 )]


            # Gather impact velocity info
        bins_impact_vs = np.arange(-2, 2.1, 0.1)
        
        avg_impact_vs_flat = np.zeros(len(bins_impact_vs)-1); avg_impact_vs_crater = np.zeros(len(bins_impact_vs)-1)
        std_impact_vs_flat = np.zeros(len(bins_impact_vs)-1); std_impact_vs_crater = np.zeros(len(bins_impact_vs)-1)
        pos_impact_vs_flat = np.zeros(len(bins_impact_vs)-1); pos_impact_vs_crater = np.zeros(len(bins_impact_vs)-1)

        impact_v = np.asarray(flat_results.impact_v)                      ; impact_v_c = np.asarray( df_impact_v['impact_v']  )
        impact_x = np.asarray(flat_results.impact_x) / flat_profile.width ; impact_x_c = np.asarray( df_impact_v['norm_impact_x']  ) # crater impact x's are already normalized

        for i in range(len(avg_impact_vs_flat)):
            pos_impact_vs_flat[i] = np.mean(bins_impact_vs[i:i+2]); pos_impact_vs_crater[i] = np.mean(bins_impact_vs[i:i+2])
            
            impact_bin   = impact_v[    np.where((impact_x >   bins_impact_vs[i]) &  
                                                 (impact_x <   bins_impact_vs[i+1])) ]
            impact_bin_c = impact_v_c[  np.where((impact_x_c > bins_impact_vs[i]) &  
                                                 (impact_x_c < bins_impact_vs[i+1])) ]
  
            
            if(len(impact_bin) != 0):
                avg_impact_vs_flat[i] = abs(np.mean(impact_bin));  avg_impact_vs_crater[i] = abs(np.mean(impact_bin_c))         
                std_impact_vs_flat[i] = np.std(impact_bin);        std_impact_vs_crater[i] = np.std(impact_bin_c)
            else:
                avg_impact_vs_flat[i] = 0;  avg_impact_vs_crater[i] = 0 
                std_impact_vs_flat[i] = 0;  std_impact_vs_crater[i] = 0     
  
    
  
        fig, axs = plt.subplots(1, figsize=(6,3))
                
        axs.hist(df_impact_v['norm_impact_x'],    bins=100, label='Crater Profiles', alpha=1.0,  facecolor='r')
        axs.hist(flat_impact_x.flatten(),         bins=100, label='Flat Profile',    alpha=0.25, facecolor='b')
        #axs.errorbar(pos_impact_vs_crater, avg_impact_vs_crater, std_impact_vs_crater*3, fmt='rs', alpha=1.0,  label='Crater Impact Velocity')
        #axs.errorbar(pos_impact_vs_flat,   avg_impact_vs_flat,   std_impact_vs_flat*3,   fmt='bo', alpha=0.25, label='Flat Impact Velocity')      

        axs.set(xlim=(-2 , 2))

        axs.legend(loc='upper right')
        axs.set_yscale('log')
        axs.set(ylim=(1,10**7))
        axs.set_title('Ejecta Impact Sites')
        axs.set(xlabel='Crater Radii')
        axs.set(ylabel='N Particles')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'Ejecta_impact_hist.png'))
        plt.show()
        

        # Make Prabal's plot
    if(True):
        prabals_plot()


    return


#######################################################
#######################################################
#######################################################


def summarize_labels(NAC_dir='NAC_DTM'):
    ''' Make a summary of the labels for all DTMs
    
    Inputs:
        NAC_dir [str] fullpath to directory containing subdirectores, each of which contains a NAC DTM and all its stuff
        
    Outputs:
        [summary.csv] file containing summary of DTM labels
    
    
    '''
    
    labelled_dirs = list_files(NAC_dir, 'a_')
    
    label_df = pd.DataFrame({'Name':         [],
                             'm/pixel':      [],
                             'ImageJ':       [],
                             'Label_Studio': [],
                             'Approved_Sims':[]})
    
        # For each labelled dir, check for 'Results.csv' (ImageJ) or Label Studio .txt file
    for this_dir in labelled_dirs:
        profile_dir  = os.path.join(this_dir, os.path.split(this_dir)[1].replace('a_', '') + '_profiles')
        track_dir    = os.path.join(profile_dir, 'track_plots_and_pkl')
        result_fname = os.path.join(profile_dir, 'sorted_results.csv')
        
        pkl_fnames = 0
        if(os.path.exists(result_fname)): 
            df = pd.read_csv(result_fname)
            
            pkl_fnames = df['pkl_fname'].loc[df['status'] == 'Y']
            pkl_fnames = len([os.path.join(track_dir, pkl_fname) for pkl_fname in pkl_fnames])
        label_df['Approved_Sims'] = pkl_fnames
        
                
            # check ImageJ
        ImageJ   = list_files(this_dir, 'Results.csv')
        N_ImageJ = 0
        if(len(ImageJ) != 0):
            N_ImageJ = len(pd.read_csv(ImageJ[0]))
        
            # Check label studio
        Label_Studio   = list_files(this_dir, '.txt')
        N_Label_Studio = 0
        if(len(Label_Studio) != 0):
            with open(Label_Studio[0].replace('.txt','.csv')) as f:
                N_Label_Studio = len(f.readlines())                
                
            # Check resolution
        lbl_fname = list_files(this_dir, '.LBL')[0]
        f = open(lbl_fname, 'r')
        lines = f.readlines()
        f.close()
            	# Set map scale from lbl file
        for i in range(len(lines)):
            	if('MAP_SCALE' in lines[i]):
                    scale = np.round(float(lines[i].split('=')[1].split('<')[0]), 0)    
            
            
            # add row to dataframe
        label_df.loc[len(label_df)] = [os.path.split(this_dir)[1],
                                       scale,
                                       N_ImageJ,
                                       N_Label_Studio]
    
    label_df.loc[len(label_df)] = ['Total',
                                  'n/a',
                                  np.sum(label_df['ImageJ']),
                                  np.sum(label_df['Label_Studio']) ] 
    
    label_df.to_csv('total_results/labels.csv', index=False)
    
    
    return
    
    
#######################################################
#######################################################
#######################################################


def prabals_plot(total_results_dir='total_results'):
    ''' Makes the statistical plots with the results of all the craters
    
    Inputs:
        total_results_dir [str] fullpath to where the results of all sims are gathered
        
    
    '''
    
    
        # Load the data using pandas with COMMA separator (not tab!)
    df = pd.read_csv(os.path.join(total_results_dir, 'compare_results.csv'))  # Default is comma-separated
    
        # Define target heights and their corresponding columns
    targets = {
        '1m': 'safe_1m',
        '5m': 'safe_5m',
        '8m': 'safe_8m',
        '16m': 'safe_16m' }
    
        # Create figure with 2 rows x 4 columns
    fig = plt.figure(figsize=(9,9))
    
        # Define d/D bins
    dD_bins     = np.arange(0.05, 0.30, 0.01)
    bin_centers = (dD_bins[:-1] + dD_bins[1:]) / 2
    
        # Color palette for each target height
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    
    ''' Overlapping density curves with transparency '''
    '''##############################################'''
    '''
    for idx, (target_name, target_col) in enumerate(targets.items()):
        ax = plt.subplot(2, 4, idx + 1)
        
            # Get d/D ratios and shadow zone sizes
        dD_values    = df['dD'].values
        shadow_zones = df[target_col].values
        
            # Filter out craters with zero or negative shadow zones
        valid_mask = shadow_zones >= 0 # CHANGED ##########################################################
        dD_filtered     = dD_values[   valid_mask]
        shadow_filtered = shadow_zones[valid_mask]
        
            # For each d/D bin, create a density curve
        for i, bin_center in enumerate(bin_centers):
                # Get data in this bin
            in_bin = (dD_filtered >= dD_bins[i]) & (dD_filtered < dD_bins[i+1])
            bin_data = shadow_filtered[in_bin]
            
            if len(bin_data) >= 3:  # Need at least 3 points for KDE
                try:
                        # Create kernel density estimate
                    kde = stats.gaussian_kde(bin_data)
                    
                        # Create y-axis range for this bin
                    y_range = np.linspace(0, shadow_filtered.max(), 200)
                    density = kde(y_range)
                    
                        # Normalize density for visualization
                    density_scaled = density * 0.005
                    
                        # Plot as a filled curve
                    ax.fill_betweenx(y_range, bin_center - density_scaled, 
                                   bin_center + density_scaled,
                                   alpha=0.3, color=colors[idx], edgecolor='none')
                except:
                    pass  # Skip if KDE fails
        
            # Add scatter points for actual data
        ax.scatter(dD_filtered, shadow_filtered, 
                  alpha=0.2, s=10, color=colors[idx], edgecolors='none')
        
            # Formatting
        ax.set_xlabel('Crater Depth/Diameter Ratio (d/D)', fontsize=11)
        ax.set_ylabel(f'Shadow Zone Size [m]', fontsize=11)
        ax.set_title(f'Option 3: Density Distribution\nTarget Height: {target_name}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.02, 0.17)
        
            # Add vertical line at d/D = 0.105 threshold
        ax.axvline(x=0.105, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                  label='d/D = 0.105 threshold')
        ax.legend(fontsize=9)
    
    '''
    ''' Percentile bands '''
    '''##################'''
    
    for idx, (target_name, target_col) in enumerate(targets.items()):
        ax = plt.subplot(2, 2, idx + 1)
        
            # Get d/D ratios and shadow zone sizes
        dD_values = df['dD'].values
        shadow_zones = df[target_col].values
        
            # Filter out craters with zero or negative shadow zones
        valid_mask = shadow_zones >= 0  ###################################################################
        dD_filtered = dD_values[valid_mask]
        shadow_filtered = shadow_zones[valid_mask]
        
            # Calculate percentiles for each bin
        valid_bins = []
        medians = []
        p25s = []
        p75s = []
        p10s = []
        p90s = []
        mins = []
        counts = []
        
        for i, bin_center in enumerate(bin_centers):
                # Get data in this bin
            in_bin = (dD_filtered >= dD_bins[i]) & (dD_filtered < dD_bins[i+1])
            bin_data = shadow_filtered[in_bin]
            
            if len(bin_data) >= 2:
                valid_bins.append(bin_center)
                medians.append(np.median(bin_data))
                p25s.append(np.percentile(bin_data, 25))
                p75s.append(np.percentile(bin_data, 75))
                p10s.append(np.percentile(bin_data, 10))
                p90s.append(np.percentile(bin_data, 90))
                mins.append(np.min(bin_data))
                counts.append(len(bin_data))
        
        valid_bins = np.array(valid_bins)
        
        if len(valid_bins) > 0:
                # Convert to arrays
            medians = np.array(medians)
            p25s = np.array(p25s)
            p75s = np.array(p75s)
            p10s = np.array(p10s)
            p90s = np.array(p90s)
            mins = np.array(mins)
            
                # Smooth the curves
            smooth_sigma = 1.0
            if len(medians) > 2:
                median_smooth = gaussian_filter1d(medians, smooth_sigma)
                p25_smooth = gaussian_filter1d(p25s, smooth_sigma)
                p75_smooth = gaussian_filter1d(p75s, smooth_sigma)
                p10_smooth = gaussian_filter1d(p10s, smooth_sigma)
                p90_smooth = gaussian_filter1d(p90s, smooth_sigma)
                min_smooth = gaussian_filter1d(mins, smooth_sigma)
            else:
                median_smooth = medians
                p25_smooth = p25s
                p75_smooth = p75s
                p10_smooth = p10s
                p90_smooth = p90s
                min_smooth = mins
            
                # Plot bands
            ax.fill_between(valid_bins, p10_smooth, p90_smooth, 
                           alpha=0.2, color=colors[idx], label='10th-90th percentile')
            ax.fill_between(valid_bins, p25_smooth, p75_smooth, 
                           alpha=0.4, color=colors[idx], label='25th-75th percentile (IQR)')
            
                # Plot lines
            ax.plot(valid_bins, median_smooth, color=colors[idx], linewidth=2.5, 
                   label='Median', marker='o', markersize=4)
            ax.plot(valid_bins, min_smooth, color='darkred', linewidth=2, 
                   linestyle='--', label='Minimum (conservative)', marker='s', markersize=3)
            
            '''
                # Add count annotations
            for i, (bin_val, count) in enumerate(zip(valid_bins, counts)):
                if i % 2 == 0:
                    ax.text(bin_val, ax.get_ylim()[1] * 0.95, f'n={count}', 
                           fontsize=8, ha='center', va='top', alpha=0.6)
            '''
        
            # Formatting
        ax.set_xlabel('Crater d/D', fontsize=11)
        ax.set_ylabel(f'Shadow Zone Size [m]', fontsize=11)
        ax.set_title(f'Target Height: {target_name}', 
                    fontsize=12, )#fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.054, 0.265)
        ax.set_ylim(0,600)
        
            # Add vertical line
        '''
        ax.axvline(x=0.175, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                  label='d/D = 0.175\n(guaranteed shadow zone)')
        '''
        
        ax.legend(fontsize=8, loc='upper left')
    
    
    plt.suptitle('Umbrella Zone Percentile Bands', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(total_results_dir ,'umbrella_zone_statistical_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nStatistical Summary Example (for 5m target):")
    print("=" * 70)
    
    dD = df['dD'].values
    safe_5m = df['safe_5m'].values
    valid = safe_5m > 0
    
    for dD_range in [(0.04, 0.06), (0.06, 0.08), (0.08, 0.10), (0.10, 0.12)]:
        in_range = valid & (dD >= dD_range[0]) & (dD < dD_range[1])
        subset = safe_5m[in_range]
        if len(subset) > 0:
            print(f"\nd/D = {dD_range[0]:.2f} to {dD_range[1]:.2f} (n={len(subset)} craters):")
            print(f"  Median shadow zone: {np.median(subset):.0f} m")
            print(f"  25th-75th percentile: {np.percentile(subset, 25):.0f} - {np.percentile(subset, 75):.0f} m")
            print(f"  Conservative (minimum): {np.min(subset):.0f} m")

    return
    
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################

        
''' Start Runfile Instructions '''
'''============================'''

if(__name__ == '__main__'):
    

            
#################################################

    
        # test PSI sim on crater, and read pkl results
    if(False):
        v0s  = np.arange(10, 2380, 10)
        degs = np.arange( 0, 20,  0.10) 

        crater_profile_fname = 'test_dir/a_NAC_DTM_AESTUUM1/NAC_DTM_AESTUUM1_profiles/crater_plots_and_pkl/IMG_NAC_DTM_AESTUUM1-X_299-Y_4401-W_58-H_40.pkl'
        simulate_PSI_with_crater(crater_profile_fname, v0s, degs, plot=True, old_tracks_img=False)
        
        
        # Run simulation on all labelled craters 
    if(False):
        
        #DTM_dir = 'NAC_DTM/a_NAC_DTM_SPARIM1'
        v0s  = np.arange(10, 2380, 10) 
        degs = np.arange(0, 20, 0.10)
        
        DTM_dirs = [#'NAC_DTM/a_NAC_DTM_A12LMAS',
                    #'NAC_DTM/a_NAC_DTM_AESTUUM1',
                    #'NAC_DTM/a_NAC_DTM_AITKEN01',       
                    #'NAC_DTM/a_NAC_DTM_GRUITHUISE1',    #
                    'NAC_DTM/a_NAC_DTM_SCHRODNGR01'      #
                    ]
        
        
        #get_crater_profiles(DTM_dirs[1])        
        for i in range(len(DTM_dirs)):
        
            simulate_PSIs_on_DTM(DTM_dirs[i], v0s, degs, overwrite=True, max_crater_diam=4000, cool_time=10, max_deg=20)
            
        # Run simulation on flat profile
    if(False):
            # 330 m; avg diam for craters
        v0s  = np.arange(10, 2380, 100) 
        degs = np.arange(0, 20, 1)
        flat_sname = 'flat_profile/flat_profile_profiles/crater_plots_and_pkl/flat_profile_test_width.pkl'
        
        simulate_PSI_with_flat(flat_sname, v0s, degs, x0s=[0], y0s=[0.9], dt=0.01, seconds=10, 
                               n_free_cores=4, width=385, plotting_width=1.5, plot=True)



        # Gather results from all DTMs
    if(False):

        track_dirs = list_files('NAC_DTM', 'a_')

        compare_results(track_dirs, gather_new=False ) 
    


    if(True):
        pkl_fname = 'zenodo_crater_pads/flat_profile/flat_profile_profiles/track_plots_and_pkl/tracks_flat_profile_385m_width.pkl'
        
        read_pkl_results(pkl_fname, plot=True)

        pkl_fname = 'zenodo_crater_pads/flat_profile/flat_profile_profiles/track_plots_and_pkl/tracks_flat_profile_1000m_width.pkl'
        
        read_pkl_results(pkl_fname, plot=True)


    

#######################################################
#######################################################
#######################################################
# OUTDATED #
#######################################################
#######################################################
#######################################################


    # Outdate because the new version makes a 'crater_profile' with the wall topography and 
    #  simply calls simulate_PSI_with_craer
def old_simulate_PSI_with_wall(wall_radius, wall_height, particle_list, ret_list, wall_width=1, n_cores=4, plotting_width=2):
    ''' Runs a PSI simulation with a list of particles on a imagined flat terrain,
    with a wall using a given distance from the landing site, and height.   
    
    '''

    ########################################    
    ########################################
    def run_particle_sim_wall(wall_radius, wall_height, particle_arr, ret_list):
        
        vx0		 = particle_arr[0]
        vy0		 = particle_arr[1]
        x0		 = particle_arr[2]
        y0		 = particle_arr[3]
        dt		 = particle_arr[4]
        seconds  = particle_arr[5]
        profile  = particle_arr[6]
        ret_list = particle_arr[7]
        
        ''' Calc where the particle lands '''
        '''==============================='''
            # quadratic equation
        t1 = (-vy0 + np.sqrt(vy0**2 - -g*2*y0)) / (2 * -g/2)
        t2 = (-vy0 - np.sqrt(vy0**2 - -g*2*y0)) / (2 * -g/2)
        
        if(t1 > 0):
            t_end = t1
        if(t2 > 0):
            t_end = t2

        ''' Calc if the particle intersects with either edge of the wall (inner or outer) '''
        '''==============================================================================='''
        
        wall_t1 = (wall_radius - x0) / vx0
        wall_y1 = (1/2 * (-g) * wall_t1**2) + (vy0 * wall_t1) + y0
        
        wall_t2 = (wall_radius - x0) / vx0
        wall_y2 = (1/2 * (-g) * wall_t2**2) + (vy0 * wall_t2) + y0

            # if the height of the particle is lower than the wall height
            #  end the track at that spot, if it isn't already ended
        if((wall_y1 < wall_height or wall_y2 < wall_height)
            and wall_t1 < t_end):
            t_end = wall_t1
        
        ''' Calc the Particle tracks for plotting '''
        '''======================================='''
        
        step_arr = np.arange(0, t_end, 0.01)    # time steps in [s]
        n_steps  = len(step_arr)
            # dont think I need these four lines
        x  = np.zeros(n_steps)
        y  = np.zeros(n_steps)
        vx = np.zeros(n_steps)
        vy = np.zeros(n_steps)

    	    # Calculate Ballistics
        x  = x0 + (vx0 * step_arr)
        y  = y0 + (vy0 * step_arr) + (-g * step_arr**2)/2.
        vx = vx0 * np.ones(n_steps)
        vy = vy0 + (-g * step_arr)
        
        theta = np.zeros(len(x))
    
        this_track = Particle_track(vx.astype(int), vy.astype(int), x.astype(int), y.astype(int), theta.astype(int))
        ret_list.append(this_track)
        
        return [vx, vy, x, y, theta]  
    ########################################
    ########################################
    
    ''' Start overall simulation '''
    '''=========================='''
    
    work_pool = mp.Pool(processes=n_cores)

    work_pool.map(run_particle_sim_array, particle_list)
    
    
    ''' Collect Results for plotting '''
    '''=============================='''
    
        # list directory structure
    crater_pkl_dir = 'NAC_DTM/a_Artificial_Pads/crater_plots_and_pkl'
    track_pkl_dir  = 'NAC_DTM/a_Artificial_Pads/track_plots_and_pkl'
    
    short_crater_name = 'ArtificialPad_R-' + str(wall_radius) + '_H-' + str(wall_height) + '_W-' + str(wall_width)
    
        # make track dir and file save names
    if(not os.path.exists(track_pkl_dir)):
        os.mkdir(track_pkl_dir)
    
    track_pkl_sname  = os.path.join(track_pkl_dir, ('tracks_' + short_crater_name))
    track_plot_sname = track_pkl_sname.replace('.pkl','.png')
    
        # Make crater (Wall) profile
        # define x and y 
    zone1 = np.arange(0, wall_radius+1, 1)
    wall1 = wall_radius + 1
    wall2 = wall1 + wall_width
    zone2 = np.arange(wall2, 100_000, 1)
    x = np.concatenate((zone1, wall1, wall2, zone2), axis=None)
    
    zone1 = np.zeros(len(zone1))
    wall1 = wall_height
    wall2 = wall_height
    zone2 = np.zeros(len(zone2))
    y = np.concatenate((zone1, wall1, wall2, zone2), axis=None)
    
    label_str = ('Artificial Berm' + '\nDistance: ' + str(wall_radius) + 
                 '\nHeight: ' + str(wall_height) + 
                 '\nWidth:  ' + str(wall_width) )
        # plot particle tracks
        #  and save tracks to pkl file
        #  and record impact point        
    plt.clf()       # w, h
    #plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(3, sharex=True, figsize=(10,8))
    axs[0].plot(x, y, 'k-', label=label_str)
        
    file_obj = open(track_pkl_sname, 'wb')
    impact_x = []
    impact_v = []
    shadow_bins = np.arange(0, plotting_width*wall_radius, 1)
    lowest_ejecta = [[] for bins in shadow_bins]
    
        # Go through each track and plot
    for track in ret_list:
            # if the track actually made it anywhere
        if(len(track.x) > 0):
            axs[0].plot(track.x, track.y, 'b-')
            pickle.dump(track, file_obj)
            
            this_impact = track.x[-1]
                # if the track impacted within the area of interest (crater_width * the plotting width)
                #  record the impact site for later histogram
            if(abs(this_impact) < wall_radius*plotting_width):
                impact_x.append(track.x[-1])
                impact_v.append(np.sqrt(track.vx[-1]**2 + track.vy[-1]**2))    
    
                
                # Go through each of the shawdow bins
            for j in range(len(shadow_bins)):
                mask = np.where((track.x >= j) & (track.x <= j+1))
                if(len(mask[0]) > 0):
                    lowest_ejecta[j].append(np.min(track.y[mask]))
                
        else:
            print('No Return:', track) 

    file_obj.close()
    
    for j in range(len(lowest_ejecta)):
        lowest_ejecta[j] = np.min(lowest_ejecta[j])
            
    lowest_ejecta = np.asarray(lowest_ejecta)
    #crater_floor  = np.asarray([calc_crater_wall(x + 0.5, crater_profile)[0] for x in shadow_bins])
    crater_floor = y[0:len(lowest_ejecta)]
    
    gaps = lowest_ejecta - crater_floor
    

    
    ''' Start Plotting '''
    '''================'''


    fig, axs = plt.subplots(3, sharex=True, figsize=(10,8))
    axs[0].plot(x, y, 'k-', label=label_str)
    
        # Plot formatting
            # Track plot (top)
    axs[0].set_title('Particle Tracks')
    axs[0].set_ylabel('[m]')
    axs[0].axis('equal')
    axs[0].set(xlim=(-wall_radius*2, wall_radius*2), ylim=(-10, wall_height*3))
    axs[0].legend(loc='center left')
    axs[0].grid()
    
            # impact position hist plot (middle)
    axs[1].set_title('Particle Impact Site')
    axs[1].set_ylabel('N Particles')
    axs[1].set_yscale('log')    
    hist_bins = axs[1].hist(impact_x, bins=np.arange(0, wall_radius*plotting_width, 1) ) 
     
            # Ejecta shadow
    axs[2].set_title('Ejecta Shadow')
    axs[2].vlines(shadow_bins, np.zeros(len(gaps)), gaps)
    axs[2].set_xlabel('[m]')
    axs[2].set_ylabel('[m]')
    axs[2].grid()
    #axs[2].axis('equal')
    
    print(hist_bins)
    
    return


#######################################################
#######################################################
#######################################################








