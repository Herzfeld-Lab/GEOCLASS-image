
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

def run_vario(ddaData, lag, windowSize, windowStep, ndir, nres, photons = False, residual = False):

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

	# Calculate eastings and northings based on lon-lat data
	eastings, northings, _, _ = utm.from_latlon(lat,lon)

	# Calculate the start and end of each window according to windowSize and windowStep
	# Note: the windows are dependent on distance along track rather than lon-lat
	windows = list(zip(np.arange(np.min(distance),np.max(distance),windowStep), np.arange(np.min(distance)+windowSize,np.max(distance)-windowStep,windowStep)))
	windows = np.array(windows)

	# initialize return array
	data_windows_all = []

	for w in range(0,len(windows)):
		# Subset the elevation data
		start = windows[w,0]
		end = windows[w,1]

		window_bool = np.logical_and(distance>=start,distance<end)
		window_data = np.array([eastings[window_bool],northings[window_bool],elevation[window_bool]]).T
		# window_data = np.array([lat[window_bool],lon[window_bool],elevation[window_bool]]).T
		if len(window_data)<5: # if we have too few datapoints in a window
			print('too few points in current window')
			continue

		data_windows_all.append(window_data)


	with Pool() as pool:
		# vario_values_ret = pool.map(compute_varios, data_windows_all)
		vario_values_ret = pool.map(partial(compute_varios, lag=lag, nres=nres, coef=coef), data_windows_all)

	return np.array(vario_values_ret)


def compute_varios(windowData, lag, nres, coef):
	"""
	window data: [east (utm), north (utm), elevation]
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
	
	vario_values = convolve1d(vario_values, coef, mode='nearest')

	return np.array(vario_values)