import os, sys, glob, yaml, warnings
import numpy as np
import cv2 as cv
import plotly.express as px
import plotly.graph_objects as go
from progress.bar import IncrementalBar
import argparse
from utils import *

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("--label_only", type=str, default=None)
parser.add_argument("--plot_only", type=str, default=None)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
warnings.filterwarnings('error')

mainDir = cfg['img_path']
dataset_path = cfg['npy_path']
classEnum = cfg['class_enum']
numClasses = len(classEnum)
winsize = cfg['window_size']
winstep = cfg['window_step']

# set up plot directory
plot_directory = mainDir + '/vario_window_plots'
if not os.path.exists(plot_directory): os.makedirs(plot_directory)

# load current data, created from createDatasetFromeDDAice.py
if dataset_path is not None:
	current_data = np.load(dataset_path, allow_pickle=True)

# Set plotting cosmetic constants
width_px = 1000 # 1200 OG
height_px = 650 # 750 OG
linesize = 2
opac = 0.5

def plot_chunks(dataTuples):

	seg = 0
	for track,elem in enumerate(dataTuples):
		print(elem)
		tag = elem[0].split('/')[-1].split('_')[0]

		ground_estimate = np.loadtxt(elem[0])
		weight_photons = np.loadtxt(elem[1])

		dist = ground_estimate[:,3]
		mindist = np.min(dist)
		maxdist = np.max(dist)

		starts = np.arange(mindist,maxdist-winsize,winstep)
		ends = np.arange(mindist+winsize,maxdist,winstep)
		start = starts[:-1]
		end = ends[:-1]

		min_dens = np.min(weight_photons[:, 5])
		max_dens = np.max(weight_photons[:, 5])
		color_range = [i for i in range(int(min_dens),int(max_dens)+1)]

		# Progress bar output to terminal for plotting status
		bar = IncrementalBar('Plotting Track {}'.format(track+1), max=len(start), suffix='%(percent)d%%')

		for num, (s,e) in enumerate(zip(start,end)):
			weighted_segment = weight_photons[np.logical_and(weight_photons[:, 4] >= s, weight_photons[:, 4] < e), :]
			ground_segment = ground_estimate[np.logical_and(ground_estimate[:, 3] >= s, ground_estimate[:, 3] < e), :]

			pltDict = {'dist': weighted_segment[:, 4], 'elevation': weighted_segment[:, 3], 'density': weighted_segment[:, 5]}
			pltDict2 = {'dist': ground_segment[:, 3], 'elevation': ground_segment[:, 2]}
			ylim = [np.min(ground_segment[:, 2]) - 10, np.max(ground_segment[:, 2]) + 10]

			fig = px.scatter(pltDict, x='dist', y='elevation', color='density', color_continuous_scale=px.colors.sequential.Turbo,
			 range_color=[int(min_dens),int(max_dens)], opacity=opac, title='track {}, chunk {} ({})'.format(track,seg,tag), range_y = ylim)
			fig.add_trace(go.Scatter(x=pltDict2['dist'], y=pltDict2['elevation'],mode='lines',line=go.scatter.Line(color='black',width=linesize),showlegend=False))
			fig.update_layout(autosize=False,width=width_px,height=height_px)
			fig.write_image(os.path.join(plot_directory, 'segment_{}.png'.format(seg)))
			seg += 1
			bar.next()
		bar.finish()


def label_images():
	# label each image in order of plot directory
	def sortFunc(str):
		return int(str.split('_')[-1].split('.')[0])

	imgs = glob.glob(plot_directory + '/*.png')
	imgs.sort(key = sortFunc)

	classArray = []
	for seg,image in enumerate(imgs):
		img = cv.imread(image)

		if img is None:
			sys.exit('Could not read the image.')

		cv.imshow("Display Window", img)
		lab = cv.waitKey(0)

		if seg % 100 == 0 and seg > 0:
			class_label_breakdown(np.array(classArray),classEnum)

		# 127 = delete key (Mac), int(chr(48)) = 0
		if lab == 127 or lab < 48 or lab >= (48 + numClasses):
			classArray.append(-1)
		else:
			classArray.append(int(chr(lab)))

	return np.array(classArray)

def get_data():

	files = glob.glob(mainDir + '/*_pass0.txt')
	files.sort()

	ge, wp = [],[]

	for file in files:

		if 'ground_estimate' in file:
			ge.append(file)
		if 'weighted_photons' in file:
			wp.append(file)

	zip_arr = zip(ge,wp)
	return list(zip_arr)


def main():

	if args.label_only is None:
		# get ground estimate / weighted photon data
		dataList = get_data()
		# plot all dda vario window chunks
		plot_chunks(dataList)

	if args.plot_only is None:
		# execute user labeling feature
		classLabels = label_images()
		# save labels to input data directory
		np.savetxt(os.path.join(mainDir, 'window_labels_auto.txt'), classLabels)
		# print class/label breakdown
		class_label_breakdown(classLabels,classEnum)

		# only do next 2 steps if there's actually a dataset, else just save txt file
		# TODO: better conditional here... didn't work last time
		if current_data:
			# add window labels to main data file
			current_data[1][:,0] = classLabels
			# save new dataset w/ valid segment labels
			np.save(dataset_path, current_data)


if __name__ == '__main__':
	main()