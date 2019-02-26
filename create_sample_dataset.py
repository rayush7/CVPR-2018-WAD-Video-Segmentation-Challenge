import numpy as np 
import pandas as pd 
import os
import string

#---------------------------------------------------------------------------------------------------------------------------------------------------

def create_sample_train_val_dataset(parent_folder_path,path_train_video_folder,num_video,sample_train_color_file,sample_train_label_file,sample_val_color_file,sample_val_label_file,split_ratio):

	video_list = os.listdir(path_train_video_folder)

	col_names = ['info1','image_name','image_label_name']

	sample_dataset_cols = ['image_path','image_label_instance']

	sample_train_df = pd.DataFrame(columns = sample_dataset_cols)
	sample_val_df = pd.DataFrame(columns = sample_dataset_cols)

	# Video Loop
	train_image_list = []
	train_label_list = []

	val_image_list = []
	val_label_list = []

	for v_id in range(num_video):

		video_name = os.path.join(path_train_video_folder,str(video_list[v_id]))
		frame_df = pd.read_csv(video_name,header=None,delimiter= ' ',names=col_names)

		num_frame = frame_df.shape[0]

		for frame_id in range(num_frame):
			frame_name = str(frame_df['image_name'][frame_id])
			frame_name = frame_name[2:31]
			print(frame_name)


			frame_label = str(frame_df['image_label_name'][frame_id])
			frame_label = frame_label[2:]
			print(frame_label)

			current_split = (frame_id+1)/num_frame
			#print(current_split)

			if split_ratio > current_split:
				train_image_list.append(frame_name)
				train_label_list.append(frame_label)
			else:
				val_image_list.append(frame_name)
				val_label_list.append(frame_label)


		#print(len(train_label_list))
		#print(len(train_image_list))

		#print(len(val_label_list))
		#print(len(val_image_list))


		sample_train_df['image_path'] = pd.Series(train_image_list)
		sample_train_df['image_label_instance'] = pd.Series(train_label_list)

		sample_val_df['image_path'] = pd.Series(val_image_list)
		sample_val_df['image_label_instance'] = pd.Series(val_label_list)


		sample_train_df.to_csv('./train_data_small.csv',index=False,header=True,sep='\t')
		sample_val_df.to_csv('./val_data_small.csv',index=False,header=True,sep='\t')

#----------------------------------------------------------------------------------------------------------------------------------------------------


def main():
	pass

if __name__ == '__main__':

	num_video = 3
	split_ratio = 0.7

	parent_folder_path = '/home/ayush/Instance_Segmentation/all'

	path_train_video_folder = os.path.join(parent_folder_path,'train_video_list')
	path_test_video = os.path.join(parent_folder_path, 'test_video_list_and_name_mapping')
	sample_train_color_file = ''
	sample_train_label_file = ''
	sample_val_color_file =  ''
	sample_val_label_file =  ''

	create_sample_train_val_dataset(parent_folder_path,path_train_video_folder,num_video,sample_train_color_file,sample_train_label_file,sample_val_color_file,sample_val_label_file,split_ratio)

	main()