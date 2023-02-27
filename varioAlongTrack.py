
import numpy as np
from scipy.ndimage.filters import convolve1d
import utm
from sklearn.metrics import pairwise_distances
import itertools
import haversine as hs
from haversine import Unit
from datetime import datetime
from multiprocessing import Pool
from functools import partial
import utils

def run_vario(ddaData, windows, lag, windowSize, windowStep, ndir, nres, photons = False, residual = False):

	# TODO: add few more comments

	###########################
	## PARAMETER DEFINITIONS ##
	###########################
	# :ddaData: --> path to dda produced ground_estimate or weighted_photon dataset
	# :lag: --> # resolution of the variogram (i.e. the spacing of the lags)
	# :windowSize: --> size of alog-track data chunk (in meters) to compute variograms on
	# :windowStep: --> magnitude of which to shift window (in meters) at every iteration of variogram computation
	# :ndir: --> number of directions to compute variograms (default = 1)
	# :nres: --> number of variogram values computed from each data window
	# :photons: --> Bool which identifies if data is photons or interpolated ground estimate
	# :residual: --> Bool to determine whether to run residual variograms or base variograms
	###########################

	lag = float(lag)
	
	# smoothing coefficients
	coef = np.array([.0625,0.25,0.375,0.25,0.625])
	coef = coef/coef.sum() # normalize

	# Load the output data from the surface detector code, likely in the following format
	# [bin_lon, bin_lat, bin_elev, bin_distance, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]
	track_data = np.loadtxt(ddaData)

	# print relevant info
	filename = ddaData.split('/')[-1]
	current = datetime.now().strftime("%H:%M:%S")
	print('Time: {}, Data: {}'.format(current,filename))

	if photons==True:
		# Format of photon data:
		# [delta_time, longitude, latitude, elevation, distance]
		lon = track_data[:,1]
		lat = track_data[:,2]
		distance = track_data[:,4] # distance along track in meters
		elevation = track_data[:,3] # corresponding photon elevation
	else:
		lon = track_data[:,0]
		lat = track_data[:,1]
		distance = track_data[:,3] # distance along track in meters
		elevation = track_data[:,2] # corresponding interpolated elevation
		density_mean = track_data[:,6]
		density_sd = track_data[:,7]


	# Calculate eastings and northings based on lon-lat data
	eastings, northings, _, _ = utm.from_latlon(lat,lon)

	# Calculate the start and end of each window according to windowSize and windowStep
	# Note: the windows are dependent on distance along track rather than lon-lat
	# windows = list(zip(np.arange(np.min(distance),np.max(distance),windowStep), np.arange(np.min(distance)+windowSize,np.max(distance)-windowStep,windowStep)))
	# windows = np.array(windows)

	# initialize return array
	data_windows_all = []

	for w in range(0,len(windows)):
		# Subset the elevation data
		start = windows[w,0]
		end = windows[w,1]

		window_bool = np.logical_and(distance>=start,distance<end)
		if photons:
			window_data = np.array([eastings[window_bool],northings[window_bool],elevation[window_bool]]).T	
		else:
			window_data = np.array([eastings[window_bool],northings[window_bool],elevation[window_bool],lon[window_bool],lat[window_bool],density_mean[window_bool],density_sd[window_bool],distance[window_bool]]).T
		# window_data = np.array([lat[window_bool],lon[window_bool],elevation[window_bool]]).T
		if len(window_data)<5: # if we have too few datapoints in a window
			print('too few points in current window')
			continue

		data_windows_all.append(window_data)

	with Pool() as pool:
		# vario_values_ret = pool.map(compute_varios, data_windows_all)
		vario_values_ret = pool.map(partial(compute_varios, lag=lag, nres=nres, coef=coef, weighted_photons=photons), data_windows_all)
	
	if photons:
		return np.array(vario_values_ret)

	with Pool() as pool2:
		density_feature = pool2.map(partial(make_density_feature, lag=lag, nres=nres), data_windows_all)

	return np.hstack((np.array(vario_values_ret), np.array(density_feature)))


def compute_varios(windowData, lag, nres, coef, weighted_photons):
	"""
	window data: [east (utm), north (utm), elevation, lon, lat]
	"""
	def get_pairs(sepDist):
		ls = []
		for i,row in enumerate(pdist):
			idxs1 = np.argwhere(row < sepDist).flatten()
			idxs2 = np.argwhere(row >= sepDist-5.0).flatten()
			temp = set(idxs2)
			idxs = [pair for pair in idxs1 if pair in temp]
			# idxs = list(set(idxs1) & set(idxs2))
			ls.append([(i,j) for j in idxs if j > i and pdist[i,j]!=0.0])

		return list(itertools.chain(*ls))

	def semivariogram(pairs):
		varSum = 0
		for (x,y) in pairs:
			varSum += ((windowData[x, 2] - windowData[y, 2])**2)
		varSum /= (2 * len(pairs))
		return varSum

	##### Functionality begins #####
	# get all pairwise distances between points in the window
	pdist = pairwise_distances(windowData[:,0:2])
	vario_values = []
	for k in range(1,nres+1):
		# get pairs of points separated by <= lag*i meters
		pairs = get_pairs(lag*k)
		if(len(pairs) == 0.0):
			vario_values.append(0.0)
			continue
		vario_values.append(semivariogram(pairs))
	
	# TODO: test w/o smoothing vario results -> maybe more precise better for indiv crevasses?
	vario_values = list(convolve1d(vario_values, coef, mode='nearest'))

	# add lon/lat values for segment identification (previous TODO: remove old code)
	# [lon_start, lat_start, lon_end, lat_end]
	# latlon = [windowData[0,3], windowData[0,4], windowData[-1,3], windowData[-1,4]]

	if not weighted_photons:
		# only include segment location info on ground estimate data (weighted photons would be redundant)
		# use midpoint of segment for segment location
		latlon = utils.get_segment_midpt_loc(windowData[0,4], windowData[0,3], windowData[-1,4], windowData[-1,3])
		full_feature = np.concatenate((latlon,vario_values))
		return full_feature
	return vario_values


def make_density_feature(windowData, lag, nres):
	dens_mean = windowData[:,5]
	dens_sd = windowData[:,6]
	distance = windowData[:,7]
	if len(dens_mean) == nres and len(dens_sd) == nres:
		return np.concatenate((dens_mean,dens_sd))
	else:
		# Non-uniform ground interpolation due to crevassing
		window_start = distance[0]
		dm, dsd = np.zeros(nres),np.zeros(nres)
		for i in range(nres):
			window_bool = np.logical_and(distance>=window_start, distance<window_start+lag)
			dm[i] = np.mean(dens_mean[window_bool])
			dsd[i] = np.mean(dens_sd[window_bool])

		return np.concatenate((dm,dsd))