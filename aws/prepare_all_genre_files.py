# coding: utf-8

import os
import glob
import pandas as pd
import shutil


# #### Get list of files in genre folder

# In[1]:

genre_list = [
#'breakbeat',
#'dancehall_ragga',
#'deep_house',
'disco',
'downtempo',
'drum_and_bass',
'dubstep_grime',
'electro_house',
'euro_dance'
]



# In[2]:

for genre_name in genre_list:
	print('Process %s.....'%genre_name)
	file_list = os.listdir('data/{}'.format(genre_name))


	# Split file name into its relevant information

	# In[3]:


	split_file_name = file_list[0].split('__')
	genre = split_file_name[0]
	track_id = split_file_name[1]
	spectro_id = split_file_name[2].split('.')[0]


	# In[4]:


	list_of_files = []
	for f in file_list:
		split_f = f.split('__')
		genre = split_f[0]
		track_id = split_f[1]
		spectro_id = split_f[2].split('.')[0]
		file_name = f
		
		track_dict = {
			'file_name':file_name,
			'track_id':track_id,
			'genre':genre,
			'spectro_id':spectro_id}
		
		list_of_files.append(track_dict)


	# In[10]:


	list_of_files[0]


	# #### Move data to DataFrame

	# In[11]:


	df = pd.DataFrame(list_of_files)


	# #### Calculate the number of spectrograms for each unique track

	# In[13]:


	df.sample(5)


	# In[14]:


	num_files = df.groupby('track_id')['file_name'].count().reset_index()
	num_files.columns = ['track_id','num_spectro']


	# In[15]:


	num_files.head()


	# #### Work out the number of files needed in the train, validation and holdout folders

	# In[16]:


	total_num_files = num_files['num_spectro'].sum()


	# In[17]:


	train_pct = 0.65
	validation_pct = 0.25
	holdout_pct = 0.1


	# In[18]:


	train_image_threshold = int(train_pct * total_num_files)
	validation_image_threshold = int(validation_pct * total_num_files)


	# In[19]:


	tracks_dict = dict(zip(num_files['track_id'], num_files['num_spectro']))


	# #### Loop through tracks_dict and make a note of track IDs that will go into the train dataset

	# In[20]:


	count = 0
	train_ids = []
	for key, value in tracks_dict.items():
		count += value
		if count <= train_image_threshold:
			train_ids.append((key, value))


	# Then remove these track id's from the dict...

	# In[21]:


	for item in train_ids:
		tracks_dict.pop(item[0], None)


	# #### Do the same for the validation set

	# In[22]:


	count = 0
	validation_ids = []
	for key, value in tracks_dict.items():
		count += value
		if count <= validation_image_threshold:
			validation_ids.append((key, value))


	# In[23]:


	for item in validation_ids:
		tracks_dict.pop(item[0], None)


	# #### Then move the rest into the holdout set

	# In[24]:


	holdout_ids = []
	for key, value in tracks_dict.items():
		holdout_ids.append((key, value))


	# Number of tracks in each set...

	# In[26]:


	print(len(train_ids))
	print(len(validation_ids))
	print(len(holdout_ids))


	# #### Create directories for train/breakbeat, validation/breakbeat and holdout/breakbeat if they don't exist

	# In[27]:


	train_dir = 'data/train/{}'.format(genre_name)
	validation_dir = 'data/validation/{}'.format(genre_name)
	holdout_dir = 'data/holdout/{}'.format(genre_name)


	if not os.path.exists(train_dir):
		os.makedirs(train_dir)

	if not os.path.exists(validation_dir):
		os.makedirs(validation_dir)

	if not os.path.exists(holdout_dir):
		os.makedirs(holdout_dir)


	# #### Move train files from data/breakbeat to data/train/breakbeat

	# In[28]:


	for t in train_ids:
		files_to_move = list(df[df['track_id'] == t[0]]['file_name'])
		for f in files_to_move:
			src = 'data/{}/{}'.format(genre_name, f)
			dst = 'data/train/{}/{}'.format(genre_name, f)
			shutil.move(src, dst)


	# #### Move validation files from data/breakbeat to data/validation/breakbeat

	# In[29]:


	for t in validation_ids:
		files_to_move = list(df[df['track_id'] == t[0]]['file_name'])
		for f in files_to_move:
			src = 'data/{}/{}'.format(genre_name, f)
			dst = 'data/validation/{}/{}'.format(genre_name, f)
			shutil.move(src, dst)


	# #### And finally the holdout files...

	# In[30]:


	for t in holdout_ids:
		files_to_move = list(df[df['track_id'] == t[0]]['file_name'])
		for f in files_to_move:
			src = 'data/{}/{}'.format(genre_name, f)
			dst = 'data/holdout/{}/{}'.format(genre_name, f)
			shutil.move(src, dst)

