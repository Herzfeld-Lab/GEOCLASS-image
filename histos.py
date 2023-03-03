import numpy as np
import utils
from datetime import datetime

# TODO: document that vlen = lag and why

def generate_histo(data, start, end, hlen, vlen, avg_ht, v_factor):
	# horizontal_edges = np.arange(np.nanmin(data[:, 4]), np.nanmax(data[:, 4]) + hlen, hlen)
	# vertical_edges = np.arange(np.nanmin(data[:, 3]), np.nanmax(data[:, 3]) + 2 * vlen, vlen)

	# remove data dependency of histogram bin structure - uniform for every along-track window
	horizontal_edges = np.arange(start, end + hlen, hlen)
	vertical_edges = np.arange(avg_ht-v_factor, avg_ht+v_factor, vlen)
	H = np.histogram2d(data[:, 4], data[:, 3], bins=(horizontal_edges, vertical_edges))[0]
	return H


def make_histo_features(ge, wp, windows, lag, vbin_size, v_window_size, nres):

	# return: [segment_lat, segment_lon, nres along track histos...] for each window

	# print relevant info
	filename = ge.split('/')[-1]
	current = datetime.now().strftime("%H:%M:%S")
	print('Time: {}, Data: {}'.format(current,filename))

	wp_data = np.loadtxt(wp)
	ge_data = np.loadtxt(ge)
	distance = ge_data[:,3]
	elevation = ge_data[:,2]
	lon = ge_data[:,0]
	lat = ge_data[:,1]

	histos,latlon = [],[]
	for w in range(len(windows)):
		start, end = windows[w,0], windows[w,1]
		window_bool = np.logical_and(distance>=start,distance<end)

		# used to center vertical window for histogram
		avg_ht = np.mean(elevation[window_bool])

		wp_window_bool = np.logical_and(wp_data[:,4]>=start, wp_data[:,4]<end)
		# each histogram should be of shape (nres, (v_window_size/vbin_size)-1)
		histo = generate_histo(wp_data[wp_window_bool], start=start, end=end, hlen=lag, vlen=vbin_size,
								avg_ht=int(avg_ht), v_factor=int(v_window_size/2))
		
		if histo.shape != (nres, int(v_window_size/vbin_size)-1):
			print(histo.shape)
			print(w)
			print('Problem w/ Histogram feature dimensions... exiting...')
			return None
		
		histos.append(histo)

		# use midpoint lat/lon of segment to identify location
		win_lon, win_lat = lon[window_bool], lat[window_bool]
		latlon.append(utils.get_segment_midpt_loc(win_lat[0], win_lon[0], win_lat[-1], win_lon[-1]))

	return np.array(latlon), np.array(histos)