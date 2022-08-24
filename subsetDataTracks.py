import numpy as np
import os, argparse

"""
TODO: fix comments / description
This is a script that takes in a DDA result: ground_estimate or weighted_photons and subsets 
them based on along-track distances in order to use them for train/test data for DDA Classification.

Goal: subset track over crevassed areas only to eliminate tons of undisturbed snow segments 

In Terminal: 
python3 subsetDataTracks.py <data> -s <start distance> -e <end distance>

<data> needs to be path to either ground_estimate or weighted_photons file output from DDA-ice-1
-s and -e are start/end distances (along-track) in meters
"""

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str) # directory path containing all DDA results to use
args = parser.parse_args()

dest = '/Users/adamhayes/workspace/train_test_data/negri-early-late-train'
if not os.path.exists(dest): os.makedirs(dest)

if 'early' in args.dir:
	tag = 'EARLY'
elif 'late' in args.dir:
	tag = 'LATE'
elif 'mid' in args.dir:
	tag = 'MID'
else:
	raise NameError('Only able to handle early / late / mid Negri results...')


results = os.listdir(args.dir)
for i,res in enumerate(results):
	if '.DS_Store' in res: continue
	path = os.path.join(args.dir, res)
	crevRange = np.loadtxt(os.path.join(path, 'crev_range.txt'))
	start,end = int(crevRange[0]), int(crevRange[1])
	
	ge = np.loadtxt(os.path.join(path, 'ground_estimate_pass0.txt'))
	wp = np.loadtxt(os.path.join(path, 'weighted_photons_pass0.txt'))

	geTemp = ge[ge[:,3] >= start]
	wpTemp = wp[wp[:,4] >= start]

	geFinal = geTemp[geTemp[:,3] <= end]
	wpFinal = wpTemp[wpTemp[:,4] <= end]

	geDest = os.path.join(dest, '{}_negri{}_ground_estimate_pass0.txt'.format(tag,i))
	wpDest = os.path.join(dest, '{}_negri{}_weighted_photons_pass0.txt'.format(tag,i))

	print('{}_negri{}: {}'.format(tag, i, res))
	np.savetxt(geDest, geFinal)
	np.savetxt(wpDest, wpFinal)





previous = False
if previous:
	# parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("data", type=str) # either ground_estimate or weighted_photons track
	parser.add_argument("-s", type=str, default=None) # distance in meters
	parser.add_argument("-e", type=str, default=None) # distance in meters
	args = parser.parse_args()

	# prompts for name and location to store result data 
	destination = input('\nWhere would you like to put the result (full or relative file path)? ')
	name = input('\nWhat would you like to name this file (do not include file extension)? ')
	name = name + '.txt'
	print('\n')

	location = os.path.join(destination,name)
	data_track = np.loadtxt(args.data)


	if args.s is None or args.e is None:
		print('You have not specified starting/ending distances, saving the full track: \n{}'.format(location))
		np.savetxt(location, data_track)
	else:
		start,end = int(args.s), int(args.e)

		if 'ground_estimate' in args.data:
			temp = data_track[data_track[:,3] >= start]
			final = temp[temp[:,3] <= end]
			dist_idx = 3
		elif 'weighted_photons' in args.data:
			temp = data_track[data_track[:,4] >= start]
			final = temp[temp[:,4] <= end]
			dist_idx = 4
		else:
			raise NameError('Only valid inputs are ground_estimate and weighted_photons')

		if np.min(final[:,dist_idx]) < start or np.max(final[:,dist_idx]) > end:
			raise ValueError('Track was not properly subsetted... Distances are off...')

		np.savetxt(location, final)




