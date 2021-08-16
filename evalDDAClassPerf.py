
import numpy as np
import os, glob
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml


def compare_labels(valid_labels, test_data, nres, num_classes = 4):

	# valid_labels = np.load(fileTrain, allow_pickle=True)[1]
	# valid_labels = valid_labels[:,-1]

	for fp in test_data:

		epoch = fp.split('/')[-1]
		print('Epoch checkpoint: ', epoch)

		pred_label_data = np.load(fp, allow_pickle=True)[1]
		pred_labels = pred_label_data[:,nres-1]
		conf = pred_label_data[:,nres]

		bools = valid_labels==pred_labels
		num_correct = bools[bools==True].shape[0]
		num_tot = valid_labels[valid_labels!=-1].shape[0]

		cm = confusion_matrix(valid_labels,pred_labels,labels=[i for i in range(num_classes)])

		print('Total Correct: {} out of {}'.format(num_correct, num_tot))

		for i in range(num_classes):

			idxs = valid_labels == i
			tot = idxs[idxs==True].shape[0]
			class_bools = valid_labels[idxs] == pred_labels[idxs]
			nc = class_bools[class_bools==True].shape[0]
			pc = nc / tot

			print('Total in Class {}: {} Total correct: {} Percent correct: {}'.format(i, tot, nc, pc))

		print('\n')
		print('Confusion Matrix: ')
		df_cm = pd.DataFrame(cm, index = [i for i in "0123"],
				  columns = [i for i in "0123"])
		print(df_cm)
		# plt.figure(figsize = (10,7))
		# sn.heatmap(df_cm, annot=True)
		# plt.show()
		print('\n')



def main():

	# Parse command line flags
	parser = argparse.ArgumentParser()
	parser.add_argument("config", type=str)
	parser.add_argument("--labels", type=str, default=None)
	args = parser.parse_args()

	if args.labels is None:
		raise Exception('Must pass in a labeled test run of form: --labels Output/OUTPUT_NAME/labels')

	# Read config file
	with open(args.config, 'r') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

	npy_data = cfg['npy_path']
	nres = cfg['nres']

	dataset = np.load(npy_data, allow_pickle=True)[1]
	base_labels = dataset[:,nres-1]
	test_data = glob.glob(args.labels + '/*.npy')

	compare_labels(base_labels,test_data,nres)









if __name__ == '__main__':
	main()


