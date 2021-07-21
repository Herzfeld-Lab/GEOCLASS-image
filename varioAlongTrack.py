'''
This is a wrapper for the fortran vario function
'''

import numpy as np
from optparse import OptionParser
import os
import subprocess as sp
from shutil import rmtree
from scipy import vectorize
from scipy.ndimage.filters import convolve1d
import utm
import glob

def run_vario(ddaData, dataPath, step, winsize, winstep, nvar, ndir):

	###########################
	# SET CONSTANT PARAMETERS #
	###########################
	window_size = winsize
	window_step = winstep
	ground_filename = ddaData
	icom1 = 'idk'
	step = float(step) # resolution of the variogram (i.e. the spacing of the lags)
	name = 'elevation'
	nres = int(0.8*window_size/step) # Number of results to calculate (depends on window size and step size)
	residual = False
	photons = False
	###########################

	# Load the output data from the surface detector code, likely in the following format
	# [bin_lon, bin_lat, bin_elev, bin_distance, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]
	ground_data = np.loadtxt(ground_filename)

	if photons==True:
		print('Treating data as weighted photon output rather than interpolated surface estimate.')
		# Format of photon data:
		# [delta_time, longitude, latitude, elevation, distance]
		delta_time = ground_data[:,0]
		lon = ground_data[:,1]
		lat = ground_data[:,2]
		distance = ground_data[:,4] # distance along track in meters
		elevation = ground_data[:,3] # corresponding photon elevation
	elif photons == 'UTM':
		X = ground_data[:, 0]
		Y = ground_data[:, 1]
		elevation = ground_data[:, 3]
	else:
		lon = ground_data[:,0]
		lat = ground_data[:,1]
		distance = ground_data[:,3] # distance along track in meters
		elevation = ground_data[:,2] # corresponding interpolated elevation
		delta_time = np.empty(np.shape(lon))
		delta_time[:] = np.nan

	# Calculate eastings and northings based on lon-lat data
	eastings, northings, _, _ = utm.from_latlon(lat,lon)

	# Calculate the start and end of each window according to window_size and window_step
	# Note: the windows are dependent on distance along track rather than lon-lat
	windows = list(zip(np.arange(np.min(distance),np.max(distance),window_step), np.arange(np.min(distance)+window_size,np.max(distance)-window_step,window_step)))
	windows = np.array(windows)

	# initialize fillable arrays
	vario_values_ret = np.zeros((len(windows),47))
	parameters = np.zeros((len(windows),13))
	# We will fill parameters with [lon, lat, distance, delta_time, utm_e, utm_n, pond, p1, p2, mindist, hdiff]
	# vario_value_ret gets variogram at each iteration

	for w in range(0,len(windows)):
		# Subset the elevation data
		start = windows[w,0]
		end = windows[w,1]
		window_bool = np.logical_and(distance>=start,distance<end)
		window_data = np.array([eastings[window_bool],northings[window_bool],elevation[window_bool]]).T

		if len(window_data)<5: # if we have too few datapoints in a window
			print('too few points in current window')
			continue

		# Save the windowed data for vario to read
		np.savetxt(os.path.join(dataPath, 'window_data.dat'), window_data, fmt='%f')

		# Create invario.dat file and specify where output should be saved
		# vario_outfile = os.path.join(dataPath, 'win_{}_{}.vario'.format(start, end))
		vario_outfile = os.path.join(dataPath, 'win_{}.vario'.format(w))
		vario_infile = os.path.join(dataPath, 'window_data.dat')
		write_invario(vario_infile, vario_outfile, icom1, ground_filename, nvar, ndir, step, nres, name)

		# Call vario
		# the subprocess module calls vario and pipes the output to vario_process
		vario_process = sp.Popen('./vario', shell=False, stderr=sp.PIPE, stdout=sp.PIPE)
		out = vario_process.communicate()[1]

		# Retrieve output from vario_out.dat
		try:
			vario_results = np.loadtxt(vario_outfile)
		except ValueError as e:
			print('Vario results error: {}'.format(e))
			print('failed to load a vario_outfile, probably because there were too many points contributing to a single vario value and numpy got confused')

		# Delete 'vario_outfile' after it is read in
		# os.remove(vario_outfile)

		# col 1 - lp2, step number
		# col 2 - distclass, distance to center of class
		# col 3 - m1, mean
		# col 4 - m2, variogram
		# col 5 - m3, residual variogram
		# col 6 - dismoy, average distance of pairs used in class
		# col 7 - distot, number of pairs used in class

		if len(vario_results.shape)<2 or vario_results.shape[0]<5:  # if there were fewer than 5 variogram values calculated
			print('too few vario results to use')
			continue


		if residual==True: # residual variogram
			vario_values = vario_results[:,4]
		else: # variogram
			vario_values = vario_results[:,3]

		########################################################################################################################
		if vario_values.shape[0] == 47:
			vario_values_ret[w] = vario_values
		lags = vario_results[:,1]

		# SMOOTHING of variogram values with linear filter
		coef = np.array([.0625,0.25,0.375,0.25,0.625])
		coef = coef/coef.sum() # normalize
		vario_values = convolve1d(vario_values, coef, mode='nearest')

		pond = np.max(vario_values)
		pond_lag = lags[np.argmax(vario_values)]

		# Calculating P1 and P2
		for i in range(len(vario_values)-1): # searching for first maximum
			max1 = vario_values[i]
			hmax1 = lags[i]
			if vario_values[i+1]<vario_values[i]: # once found, go on and look for the min
				for j in range(i+1,len(vario_values)-1): # search for the first minimum after the first max
					min1 = vario_values[j]
					hmin1 = lags[j]
					if vario_values[j+1]>vario_values[j]: # if found, break. If we never find it, it's the last value (the smallest)
						break
				break

		try:
			# Calculating parameters
			if (hmin1 - hmax1) != 0:
				p1 = (max1-min1)/(hmin1-hmax1)
			else:
				p1 = 0
			p2 = (max1-min1)/max1
			mindist = hmin1
			hdiff = hmin1-hmax1
			nugget = vario_values[0]
			photon_density = float(len(window_data))/window_size
		except UnboundLocalError:  # Happens when min1 is not defined. Something in the fortran code causes this
			continue


		# Append parameters with 
		#[lon_bar, lat_bar, utm_east, utm_north, dist_bar, delta_time_bar, pond, p1, p2, mindist, hdiff, nugget, photon_density]
		lon_bar, lat_bar = np.mean(lon[window_bool]), np.mean(lat[window_bool]) 
		dist_bar, delta_time_bar = np.mean(distance[window_bool]), np.mean(delta_time[window_bool])
		utm_east_bar, utm_north_bar = np.mean(eastings[window_bool]), np.mean(northings[window_bool])
		parameters[w] = np.array([lon_bar, lat_bar, dist_bar, delta_time_bar, utm_east_bar, utm_north_bar, pond, p1, p2, mindist, hdiff, nugget, photon_density])

	if os.path.isfile(dataPath+'/window_data.dat'): os.remove(dataPath+'/window_data.dat')
	if os.path.isfile('invario.dat'): os.remove('invario.dat')
	if os.path.isfile('fort.61'): os.remove('fort.61')

	print(parameters.shape)
	print(vario_values_ret.shape)

	return vario_values_ret


	

'''
Function that takes a bunch of parameters and writes the invario.dat file
'''
def write_invario(infile, outfile, icom1='variopy', icom2='variopy', nvar=1, ndir=1,
			step=0, nres=40, name='var', speto=0, alpha=0,
			core=0, ilog=0, iacm=0, bornl=0, bornu=0,
			ychel=0, xchel=0, yinf=0, k1=0):
	invario = '%s\n%s\n%s,%s,%s,%s\n%s\n%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n%s\n%s\n' % (icom1, icom2,
			nvar, ndir, step, nres, name, speto, alpha, core, ilog, iacm,
			bornl, bornu, ychel, xchel, yinf, k1, infile, outfile)
	#invario += '\n'.join(['data.var%d' % i for i in xrange(ndir)])+'\n'
	f = open('invario.dat', 'w')
	f.write(invario)
	f.close()


