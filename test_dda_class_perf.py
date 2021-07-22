
import numpy as np
import os, glob


def compare_labels(fileTrain, fileValid, num_classes = 4):

	train_labels = np.load(fileTrain, allow_pickle=True)[1]
	train_labels = train_labels[:,-1]

	for fp in fileValid:

		epoch = fp.split('/')[-1]
		print('Epoch checkpoint: ', epoch)

		valid_labels = np.load(fp, allow_pickle=True)[1]
		valid_labels = valid_labels[:,4]

		bools = train_labels==valid_labels
		num_correct = bools[bools==True].shape[0]

		print('Total Correct: {} out of {}'.format(num_correct,len(train_labels)))

		for i in range(num_classes):

			idxs = train_labels == i
			tot = idxs[idxs==True].shape[0]
			class_bools = train_labels[idxs] == valid_labels[idxs]
			nc = class_bools[class_bools==True].shape[0]
			pc = nc / tot

			print('Total in Class {}: {} Total correct: {} Percent correct: {}'.format(i, tot, nc, pc))

		print('\n')

def main():

	base_labels = '/Users/adamhayes/workspace/NN_Class/Config/dda_test_negri_valid/dda_test_negri_valid_still_testing.npy'
	valid_label_dir = '/Users/adamhayes/workspace/NN_Class/Output/dda_test_negri_21-07-2021_16:14/labels'
	valid_data = glob.glob(valid_label_dir + '/*.npy')
	valid_paths = []
	for file in valid_data:
		path_full = valid_label_dir + file
		valid_paths.append(path_full)

	compare_labels(base_labels,valid_paths)











if __name__ == '__main__':
	main()


"""
	idxs0 = train_labels == 0
	idxs1 = train_labels == 1
	idxs2 = train_labels == 2
	idxs3 = train_labels == 3

	bools0 = train_labels[idxs0] == valid_labels[idxs0]
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