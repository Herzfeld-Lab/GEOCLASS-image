import utils
from varioAlongTrack import *
from histos import *
import yaml
import warnings
import argparse


def main():
	# Parse command line flags
	parser = argparse.ArgumentParser()
	parser.add_argument("config", type=str)
	args = parser.parse_args()

	# Read config file
	with open(args.config, 'r') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	warnings.filterwarnings('error')

	# Load config parameters
	topDir = cfg['img_path']
	classEnum = cfg['class_enum']
	chunkSize = cfg['track_chunk_size']
	lag = cfg['lag_dist']
	winsize = cfg['window_size']
	winstep = cfg['window_step']
	ndir = cfg['num_dir']
	nres = int(winsize / lag)

	# TODO: make these config params
	histo_approach = True
	vbin_size = 0.2

	print('**** Loading DDA-ice Data ****')

	bin_labels = None
	ground_est, weight_photons = [],[]
	ddaOuts = utils.get_dda_paths(topDir)

	def get_windows(ge):
		# Calculate the start and end of each window according to windowSize and windowStep
		# Note: the windows are dependent on distance along track rather than lon-lat
		distance = np.loadtxt(ge)[:,3]
		windows = list(zip(np.arange(np.min(distance),np.max(distance),winstep), np.arange(np.min(distance)+winsize,np.max(distance)-winstep,winstep)))
		return np.array(windows)

	for num, data_path in enumerate(ddaOuts):

		if 'weighted' in data_path:
			weight_photons.append(data_path)

		if 'ground' in data_path:
			ground_est.append(data_path)

		# TODO: need better option here to check with user every time
		# b/c problem when old labels are there, but you want to make new ones
		if 'window_labels_auto' in data_path:
			bin_labels = np.loadtxt(data_path)

	info = {'data_location': topDir,
			'num_tracks': len(ground_est),
			'class_enumeration': classEnum}

	if not histo_approach:
		# compute variograms and pond parameters
		print('**** Computing Variograms ****')
		feat_type = 'variogram'

		# check for multiple ground estimate files
		if len(ground_est) == 1:
			ground_est = ground_est[0]
			weight_photons = weight_photons[0]
			vario_data_ge = run_vario(ground_est, lag, winsize, winstep, ndir, nres)
			vario_data_wp = run_vario(weight_photons, lag, winsize, winstep, ndir, nres, photons=True)
		else:
			# compute variograms for each track individually, then combine results
			ge_dat, wp_dat = [],[]
			for (ge,wp) in zip(ground_est, weight_photons):
				windows = get_windows(ge)
				ge_dat.append(run_vario(ge, windows, lag, winsize, winstep, ndir, nres))
				wp_dat.append(run_vario(wp, windows, lag, winsize, winstep, ndir, nres, photons=True))
				if np.loadtxt(ge)[0,6] != ge_dat[-1][0,32] or np.loadtxt(ge)[0,7] != ge_dat[-1][0,62]:
					print("problem")
				print(len(ge_dat[-1]))
				print(len(wp_dat[-1]))
			vario_data_ge = np.vstack((ge_dat))
			vario_data_wp = np.vstack((wp_dat))

		print('**** Saving Dataset ****')

		if bin_labels is None:
			print("No pre-loaded LABELS provided, initializing labels to -1...")
			bin_labels = np.full(shape=(vario_data_ge.shape[0],1), fill_value=-1)
		else:
			print("Adding pre-loaded LABELS from data directory...")
		confidence = np.full(shape=(vario_data_ge.shape[0],1), fill_value=0)

		# format: [labels, confidence, segment_lat, segment_lon, vario(ground_estimate)..., vario(weighted_photons)...]
		feature_data = np.c_[bin_labels,confidence,vario_data_ge,vario_data_wp]

	else:
		# compute histogram features
		print('**** Computing Histograms ****')
		feat_type = 'histogram'
		
	
	print('Shape of {} data: {}'.format(feat_type, feature_data.shape))

	full_data_array = np.array([info, feature_data], dtype='object')

	dataset_path = args.config[:-7] + '_still_testing'

	cfg['npy_path'] = dataset_path + '.npy'
	cfg['nres'] = nres

	np.save(dataset_path, full_data_array)

	f = open(args.config, 'w')
	f.write(utils.generate_config_adam(cfg))
	f.close()




if __name__ == '__main__':
    main()













