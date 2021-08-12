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
nres = cfg['nres']
classEnum = cfg['class_enum']

# set up plot directory
plot_directory = mainDir + '/vario_window_plots'
if not os.path.exists(plot_directory): os.makedirs(plot_directory)

# load current data, created from createDatasetFromeDDAice.py
current_data = np.load(dataset_path, allow_pickle=True)

# Set plotting cosmetic constants
winsize = 150
winstep = 50
width_px = 1500
height_px = 750
linesize = 3
opac = 0.6

def plot_chunks(dataTuples):

	seg = 0
	for track,elem in enumerate(dataTuples):
		# print(elem[0].split('/')[-1].split('_')[0])
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

		# Progress bar output to terminal for plotting status
		bar = IncrementalBar('Plotting Track {}'.format(track+1), max=len(start), suffix='%(percent)d%%')

		for num, (s,e) in enumerate(zip(start,end)):
			weighted_segment = weight_photons[np.logical_and(weight_photons[:, 4] >= s, weight_photons[:, 4] < e), :]
			ground_segment = ground_estimate[np.logical_and(ground_estimate[:, 3] >= s, ground_estimate[:, 3] < e), :]

			pltDict = {'dist': weighted_segment[:, 4], 'elevation': weighted_segment[:, 3], 'density': weighted_segment[:, 5]}
			pltDict2 = {'dist': ground_segment[:, 3], 'elevation': ground_segment[:, 2]}
			ylim = [np.min(ground_segment[:, 2]) - 10, np.max(ground_segment[:, 2]) + 10]

			f1 = px.scatter(pltDict, x='dist', y='elevation', color='density', opacity=opac)
			f2 = px.line(pltDict2, x='dist', y='elevation')
			f2.update_traces(line=dict(color="Black", width=linesize))
			lo = go.Layout(title=go.layout.Title(text='track {}, chunk {}'.format(track,seg)))
			fig = go.Figure(data=f1.data + f2.data, layout_yaxis_range=ylim, layout=lo)
			fig.update_layout(autosize=False,width=width_px,height=height_px)
			# fig.data = fig.data[::-1]
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

		if lab == 127:
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
		# add window labels to main data file
		current_data[1][:,nres-1] = classLabels
		# save new dataset w/ valid segment labels
		np.save(dataset_path, current_data)


if __name__ == '__main__':
	main()