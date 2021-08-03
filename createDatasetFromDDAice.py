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
nres = int(0.8*winsize/lag)

print('**** Loading DDA-ice Data ****')

transforms = None
bin_labels = None
ground_est0 = []
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
			ground_est0.append(data_path)
		else:
			ground_est1 = data_path

	if 'window_labels' in data_path:
		bin_labels = np.loadtxt(data_path)


info = {'filename': ddaOuts,
		'ground_files': ground_est0,
		'transform': transforms,
		'class_enumeration': classEnum}


# compute variograms and pond parameters
print('**** Computing Variograms ****')

split_path = args.config.split('/')
dir_path = '/'.join([split_path[0],split_path[1]])

# check for multiple ground estimate files
if len(ground_est0) == 1:
	ground_est0 = ground_est0[0]
	vario_data = run_vario(ground_est0, dir_path, lag, winsize, winstep, nvar, ndir, nres)
else:
	# compute variograms for each track individually, then combine results
	vario_dat = []
	for data in ground_est0:
		vario_dat.append(run_vario(data, dir_path, lag, winsize, winstep, nvar, ndir, nres))
	vario_data = np.vstack((vario_dat))


bin_labels = np.full(shape=(vario_data.shape[0],1), fill_value=-1)
vario_data = np.c_[vario_data,bin_labels]

subset = False
if subset == True:
	vario_data_ls = []
	for i in range(len(bin_labels)):
		if bin_labels[i] == 0:
			n = np.random.uniform()
			if n > 0.83:
				vario_data_ls.append(vario_data[i])

		elif bin_labels[i] == 3:
			n2 = np.random.uniform()
			if n2 > 0.6:
				vario_data_ls.append(vario_data[i])

		else:
			vario_data_ls.append(vario_data[i])

	vario_data = np.array(vario_data_ls)


print('**** Saving Dataset ****')

full_data_array = np.array([info, vario_data], dtype='object')

dataset_path = args.config[:-7] + '_still_testing'

cfg['npy_path'] = dataset_path + '.npy'
cfg['nres'] = nres

np.save(dataset_path, full_data_array)

f = open(args.config, 'w')
f.write(generate_config_adam(cfg))
f.close() 













