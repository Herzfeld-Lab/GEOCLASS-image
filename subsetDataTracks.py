import numpy as np
import os, argparse

"""
This is a script that takes in a DDA result: ground_estimate or weighted_photons 
and subsets them based on distances in order to use them for train/test data for DDA Classification.

This allows us to include only the crevassed areas of a data track for classification purposes.

Note: a single full track will include enough undisturbed snow segments to suffice, subset all other tracks
"""

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("data", type=str) # either ground_estimate or weighted_photons track
parser.add_argument("-s", type=str, default=None)
parser.add_argument("-e", type=str, default=None)
args = parser.parse_args()

# prompts for name and location to store result data 
destination = input('\nWhere would you like to put this (full or relative file path)? ')
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
	elif 'weighted_photons' in args.data:
		temp = data_track[data_track[:,4] >= start]
		final = temp[temp[:,4] <= end]
	else:
		raise NameError('Only valid inputs are ground_estimate and weighted_photons')

	if np.min(final[:,3]) < start or np.max(final[:,3]) > end:
		raise ValueError('Track was not properly subsetted... Distances are off...')

	np.savetxt(location, final)




