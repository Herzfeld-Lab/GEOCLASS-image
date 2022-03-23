'''
This is a wrapper for the fortran vario function
'''

import numpy as np
import os, glob, re
from scipy.ndimage.filters import convolve1d
import utm
from sklearn.metrics import pairwise_distances
import itertools
import haversine as hs
from haversine import Unit

def run_vario(ddaData, lag, windowSize, windowStep, ndir, nvar = 1, photons = False, residual = False):

	# TODO: clean up unnecessary code
	# TODO: write parameter descriptions 

	###########################
	## PARAMETER DEFINITIONS ##
	
	# ddaData = path to dda produced ground_estimate or weighted_photon dataset

	###########################



	lag = float(lag) # resolution of the variogram (i.e. the spacing of the lags)
	nres = int(windowSize / lag) # Number of results to calculate (depends on window size and lag size)
	###########################

	# Load the output data from the surface detector code, likely in the following format
	# [bin_lon, bin_lat, bin_elev, bin_distance, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]
	ground_data = np.loadtxt(ddaData)

	if photons==True:
		print('Treating data as weighted photon output rather than interpolated surface estimate.')
		# Format of photon data:
		# [delta_time, longitude, latitude, elevation, distance]
		delta_time = ground_data[:,0]
		lon = ground_data[:,1]
		lat = ground_data[:,2]
		distance = ground_data[:,4] # distance along track in meters
		elevation = ground_data[:,3] # corresponding photon elevation
	else:
		lon = ground_data[:,0]
		lat = ground_data[:,1]
		distance = ground_data[:,3] # distance along track in meters
		elevation = ground_data[:,2] # corresponding interpolated elevation
		delta_time = ground_data[:,4]
		density = ground_data[:,6]

	# Calculate eastings and northings based on lon-lat data
	eastings, northings, _, _ = utm.from_latlon(lat,lon)

	# Calculate the start and end of each window according to windowSize and windowStep
	# Note: the windows are dependent on distance along track rather than lon-lat
	windows = list(zip(np.arange(np.min(distance),np.max(distance),windowStep), np.arange(np.min(distance)+windowSize,np.max(distance)-windowStep,windowStep)))
	windows = np.array(windows)

	winsize_bins = int(windowSize / lag)
	stepsize_bins = int(windowStep / lag)

	# initialize fillable arrays
	vario_values_ret = np.zeros((len(windows),nres-1))
	parameters = np.zeros((len(windows),13))
	# We will fill parameters with [lon, lat, distance, delta_time, utm_e, utm_n, pond, p1, p2, mindist, hdiff]
	# vario_value_ret gets variogram at each iteration

	for w in range(0,len(windows)):
		# Subset the elevation data
		start = windows[w,0]
		end = windows[w,1]
		
		window_bool = np.logical_and(distance>=start,distance<end)
		# window_data = np.array([eastings[window_bool],northings[window_bool],elevation[window_bool]]).T
		window_data = np.array([lat[window_bool],lon[window_bool],elevation[window_bool]]).T
		if len(window_data)<5: # if we have too few datapoints in a window
			print('too few points in current window')
			continue

		vario_results = compute_varios(window_data, ndir, lag, nres, nvar)
		print(vario_results)
		break

		# col 1 - lp2, step number
		# col 2 - distclass, distance to center of class
		# col 3 - m1, mean
		# col 4 - m2, variogram
		# col 5 - m3, residual variogram
		# col 6 - dismoy, average distance of pairs used in class
		# col 7 - distot, number of pairs used in class

		if vario_results.shape[0] != 23:
			print(vario_results.shape)

		if len(vario_results.shape)<2 or vario_results.shape[0]<5:  # if there were fewer than 5 variogram values calculated
			print('too few vario results to use')
			continue


		if residual==True: # residual variogram
			vario_values = vario_results[:,4]
		else: # variogram
			vario_values = vario_results[:,3]

		lags = vario_results[:,1]

		# SMOOTHING of variogram values with linear filter
		coef = np.array([.0625,0.25,0.375,0.25,0.625])
		coef = coef/coef.sum() # normalize
		vario_values = convolve1d(vario_values, coef, mode='nearest')

		if vario_values.shape[0] == nres-1:
			vario_values_ret[w] = vario_values
		else:
			vario_values_ret[w,:vario_values.shape[0]] = vario_values 



	return np.array(vario_values_ret)



def compute_varios(windowData, ndir, lag, nres, nvar):

	"""
	window data: [east (utm), north (utm), elevation]
	"""
	def get_pairs(sepDist):
		ls = []
		for i,row in enumerate(pdist):
			# print(row)
			idxs = np.argwhere(row < sepDist).flatten()
			ls.append([(i,j) for j in idxs if j > i])

		return list(itertools.chain(*ls))

	def semivariogram():
		varSum = 0
		for (x,y) in pairs:
			varSum += (windowData[x, 2] - windowData[y, 2])**2
		varSum /= (2 * len(pairs))
		return varSum

	def haversine_dist(arr1, arr2):
		# print(arr1)
		return hs.haversine(arr1,arr2, unit=Unit.METERS)


	##########################################################
	################  functionality begins  ##################
	##########################################################
	# get all pairwise distances between points in the window
	pdist = pairwise_distances(windowData[:,0:2], metric=haversine_dist)
	# print(pdist)
	vario_values = []
	for i in range(1,nres+1):
		# get pairs of points separated by <= lag*i meters
		pairs = get_pairs(lag*i)
		vario_values.append(semivariogram())


	


	return vario_values


	
