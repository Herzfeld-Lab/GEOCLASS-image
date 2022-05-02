from utils import *
from varioAlongTrack import *
import yaml
import warnings
import utm


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
nvar = cfg['num_var']
ndir = cfg['num_dir']
nres = int(winsize / lag)

print('**** Loading DDA-ice Data ****')

transforms = None
bin_labels = None
ground_est0, weight_photons0 = [],[]
ddaOuts = get_dda_paths(topDir)

for num, data_path in enumerate(ddaOuts):

	if 'class' in data_path:
		if '0' in data_path:
			photon_class0 = data_path
		else:
			photon_class1 = data_path

	if 'weighted' in data_path:
		if '0' in data_path:
			weight_photons0.append(data_path)
		else:
			weight_photons1 = data_path

	if 'ground' in data_path:
		if '0' in data_path:
			ground_est0.append(data_path)
		else:
			ground_est1 = data_path

	if 'window_labels_auto' in data_path:
		bin_labels = np.loadtxt(data_path)

info = {'data_location': topDir,
		'num_tracks': len(ground_est0),
		'class_enumeration': classEnum}


# compute variograms and pond parameters
print('**** Computing Variograms ****')

split_path = args.config.split('/')
# dir_path = '/'.join([split_path[0],split_path[1]])

# check for multiple ground estimate files
if len(ground_est0) == 1:
	ground_est0 = ground_est0[0]
	weight_photons0 = weight_photons0[0]
	vario_data_ge = run_vario(ground_est0, lag, winsize, winstep, ndir, nres)
	vario_data_wp = run_vario(weight_photons0, lag, winsize, winstep, ndir, nres, photons=True)
else:
	# compute variograms for each track individually, then combine results
	ge_dat, wp_dat = [],[]
	for data in ground_est0:
		ge_dat.append(run_vario(data, lag, winsize, winstep, ndir, nres))
	vario_data_ge = np.vstack((ge_dat))

	for data in weight_photons0:
		wp_dat.append(run_vario(data, lag, winsize, winstep, ndir, nres, photons=True))
	vario_data_wp = np.vstack((wp_dat))



print('**** Saving Dataset ****')

if bin_labels is None:
	bin_labels = np.full(shape=(vario_data_ge.shape[0],1), fill_value=-1)
confidence = np.full(shape=(vario_data_ge.shape[0],1), fill_value=0)

vario_data = np.c_[bin_labels,confidence,vario_data_ge,vario_data_wp]
# vario_data = np.c_[vario_data,bin_labels]

print('Shape of variogram data: {}'.format(vario_data.shape))

full_data_array = np.array([info, vario_data], dtype='object')

dataset_path = args.config[:-7] + '_still_testing'

cfg['npy_path'] = dataset_path + '.npy'
cfg['nres'] = nres

np.save(dataset_path, full_data_array)

f = open(args.config, 'w')
f.write(generate_config_adam(cfg))
f.close() 













