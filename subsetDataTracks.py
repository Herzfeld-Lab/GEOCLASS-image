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
parser.add_argument("-g","--glacier", type=str, default="none")
args = parser.parse_args()

dest = '/Users/adamhayes/workspace/train_test_data/negri-early-late-train'
if not os.path.exists(dest): os.makedirs(dest)

# glacier name tag for identifying data (i.e. negri, jak, peter)
tag = args.glacier

results = os.listdir(args.dir)
for i,res in enumerate(results):
	if '.DS_Store' in res: continue
	path = os.path.join(args.dir, res)
	if not os.path.exists(os.path.join(path, 'crev_range.txt')): continue
	crevRange = np.loadtxt(os.path.join(path, 'crev_range.txt'))
	start,end = int(crevRange[0]), int(crevRange[1])
	
	ge = np.loadtxt(os.path.join(path, 'ground_estimate_pass0.txt'))
	wp = np.loadtxt(os.path.join(path, 'weighted_photons_pass0.txt'))

	geTemp = ge[ge[:,3] >= start]
	wpTemp = wp[wp[:,4] >= start]

	geFinal = geTemp[geTemp[:,3] <= end]
	wpFinal = wpTemp[wpTemp[:,4] <= end]

	# File naming convention: <glacier><track #>_ground_estimate_pass0.txt OR <glacier><track #>_weighted_photons_pass0.txt
	# NOTE: track # is up to user to set for keeping track of records
	geDest = os.path.join(dest, '{}{}_ground_estimate_pass0.txt'.format(tag,i))
	wpDest = os.path.join(dest, '{}{}_weighted_photons_pass0.txt'.format(tag,i))

	print('{}{}: {}'.format(tag, i, res))
	np.savetxt(geDest, geFinal)
	np.savetxt(wpDest, wpFinal)

	# if os.path.exists(os.path.join(path, 'crev_range2.txt')):




