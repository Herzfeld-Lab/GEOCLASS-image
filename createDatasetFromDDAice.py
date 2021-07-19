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
step = cfg['step_size']
winsize = cfg['window_size']
winstep = cfg['window_step']
nvar = cfg['num_var']
ndir = cfg['num_dir']

print('**** Loading DDA-ice Data ****')

transforms = None
bin_labels = None
ddaOuts = get_dda_paths(topDir)

for num, data_path in enumerate(ddaOuts):

	if 'class' in data_path:
		if '0' in data_path:
			photon_class0 = data_path
		else:
			photon_class1 = data_path

	if 'weighted' in data_path:
		if '0' in data_path:
			weight_photons0 = data_path
		else:
			weight_photons1 = data_path

	if 'ground' in data_path:
		if '0' in data_path:
			ground_est0 = data_path
		else:
			ground_est1 = data_path




if bin_labels is None:
	along_track_data = ground_est0
else:
	along_track_data = np.c_[ground_est0,bin_labels]

info = {'filename': ddaOuts,
		'chunksize_m': chunkSize,
		'window_size': winsize,
		'transform': transforms,
		'class_enumeration': classEnum}

# compute variograms and pond parameters
print('**** Computing Variograms ****')
split_path = args.config.split('/')
dir_path = '/'.join([split_path[0],split_path[1]])
# pond_params, vario_data = run_vario(along_track_data, dir_path, step, winsize, winstep, nvar, ndir)
vario_data = run_vario(along_track_data, dir_path, step, winsize, winstep, nvar, ndir)
bin_labels = np.random.randint(0,3,size=(vario_data.shape[0],1))
print(vario_data.shape)
print(bin_labels.shape)
# vario_data = np.concatenate((vario_data.T,bin_labels),axis=1)
vario_data = np.c_[vario_data,bin_labels]

print('**** Saving Dataset ****')

# full_data_array = np.array([info, along_track_data, pond_params, vario_data], dtype='object')
# full_data_array = np.array([info, pond_params, vario_data], dtype='object')
full_data_array = np.array([info, vario_data], dtype='object')
print(vario_data.shape)
print(vario_data[0])

dataset_path = args.config[:-7] + '_still_testing'

cfg['npy_path'] = dataset_path + '.npy'

np.save(dataset_path, full_data_array)

f = open(args.config, 'w')
f.write(generate_config_adam(cfg))
f.close() 
















