
import numpy as np
import os, glob
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def compare_labels(fileTrain, fileValid, num_classes = 4):

	valid_labels = np.load(fileTrain, allow_pickle=True)[1]
	valid_labels = valid_labels[:,-1]

	for fp in fileValid:

		epoch = fp.split('/')[-1]
		print('Epoch checkpoint: ', epoch)

		pred_labels = np.load(fp, allow_pickle=True)[1]
		pred_labels = pred_labels[:,4]

		bools = valid_labels==pred_labels
		num_correct = bools[bools==True].shape[0]
		num_tot = valid_labels[valid_labels!=-1].shape[0]

		# valid_labels = np.ma.masked_where(valid_labels==-1,valid_labels)
		# pred_labels = np.ma.masked_where(pred_labels==-1,pred_labels)

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

	base_labels = '/Users/adamhayes/ws_home/NN_Class/Config/dda_test_both_negri_jak_valid/dda_test_both_valid_still_testing.npy'
	valid_label_dir = '/Users/adamhayes/ws_home/NN_Class/Output/dda_test_both_negri_jak_12-08-2021_16-00/labels'

	valid_data = glob.glob(valid_label_dir + '/*.npy')
	valid_paths = []
	for file in valid_data:
		path_full = file
		valid_paths.append(path_full)

	compare_labels(base_labels,valid_paths)









if __name__ == '__main__':
	main()


"""
	idxs0 = valid_labels == 0
	idxs1 = train_labels == 1
	idxs2 = train_labels == 2
	idxs3 = train_labels == 3

	bools0 = train_labels[idxs0] == pred_labels[idxs0]
	num0correct = bools0[bools0==True].shape[0]

	bools1 = train_labels[idxs1] == valid_labels[idxs1]
	num1correct = bools1[bools1==True].shape[0]

	bools2 = train_labels[idxs2] == valid_labels[idxs2]
	num2correct = bools2[bools2==True].shape[0]

	bools3 = train_labels[idxs3] == valid_labels[idxs3]
	num3correct = bools3[bools3==True].shape[0]

	tot0 = idxs0[idxs0==True].shape[0]
	tot1 = idxs1[idxs1==True].shape[0]
	tot2 = idxs2[idxs2==True].shape[0]
	tot3 = idxs3[idxs3==True].shape[0]

	perc0correct = num0correct / tot0
	perc1correct = num1correct / tot1
	perc2correct = num2correct / tot2
	perc3correct = num3correct / tot3

	print("Epoch checkpoint: ", eCheck)
	print("Total in Class 0: ", tot0, " Total correct: ", num0correct, " Percent correct: ", perc0correct)
	print("Total in Class 1: ", tot1, " Total correct: ", num1correct, " Percent correct: ", perc1correct)
	print("Total in Class 2: ", tot2, " Total correct: ", num2correct, " Percent correct: ", perc2correct)
	print("Total in Class 3: ", tot3, " Total correct: ", num3correct, " Percent correct: ", perc3correct)
	print("\n")
"""