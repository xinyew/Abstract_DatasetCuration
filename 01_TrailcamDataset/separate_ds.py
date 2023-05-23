import numpy as np
from sklearn.utils import shuffle
import csv
from PIL import Image
import os

## Define train/validate/test ratios
training_pct = 75
validation_pct = 10
test_pct = 15

global_dirext = 'tn-optimized'

## Create required files and dirs
if not os.path.exists(global_dirext):
    os.mkdir(global_dirext)
if not os.path.exists(global_dirext + '/test'):
    os.mkdir(global_dirext + '/test')
if not os.path.exists(global_dirext + '/train'):
    os.mkdir(global_dirext + '/train')
if not os.path.exists(global_dirext + '/validate'):
    os.mkdir(global_dirext + '/validate')
if not os.path.exists(global_dirext + '/test-labels.csv'):
    open(global_dirext + '/test-labels.csv', 'a').close()
if not os.path.exists(global_dirext + '/train-labels.csv'):
    open(global_dirext + '/train-labels.csv', 'a').close()
if not os.path.exists(global_dirext + '/validate-labels.csv'):
    open(global_dirext + '/validate-labels.csv', 'a').close()


## Load original dataset to be partitioned
images = np.load( 'fullds-image.npy' )
labels = np.load( 'fullds-label.npy' ).astype(dtype=int)

num_samples = len( labels )

filenames = np.array([ 'IMG_%s.png' % (i) for i in range( num_samples ) ])

## Separate for positive and negative samples
positive_indices = np.where( labels == 1 )[0]
positive_images = images[positive_indices]
positive_labels = labels[positive_indices]
positive_filenames = filenames[positive_indices]
pos_samples = len( positive_labels )

negative_indices = np.where( labels == 0 )[0]
#scale_ratio = int(1.25*pos_samples)
scale_ratio = -1
negative_images = images[negative_indices][:scale_ratio]
negative_labels = labels[negative_indices][:scale_ratio]
negative_filenames = filenames[negative_indices][:scale_ratio]
neg_samples = len( negative_labels )

pos_images_shuff, pos_labels_shuff, pos_filenames_shuff = shuffle( positive_images, positive_labels, positive_filenames, random_state=0 )
neg_images_shuff, neg_labels_shuff, neg_filenames_shuff = shuffle( negative_images, negative_labels, negative_filenames, random_state=0 )

## Compute train/validate/test lengths

pos_validation_length = int( pos_samples * validation_pct / 100 )
pos_test_length = int( pos_samples * test_pct / 100 )
pos_training_length = pos_samples - ( pos_validation_length + pos_test_length )

neg_validation_length = int( neg_samples * validation_pct / 100 )
neg_test_length = int( neg_samples * test_pct / 100 )
neg_training_length = neg_samples - ( neg_validation_length + neg_test_length )

training_length = pos_training_length + neg_training_length
validation_length = pos_validation_length + neg_validation_length
test_length = pos_test_length + neg_test_length

training_images = np.vstack( ( pos_images_shuff[:pos_training_length], neg_images_shuff[:neg_training_length] ))
training_labels = np.concatenate( ( pos_labels_shuff[:pos_training_length], neg_labels_shuff[:neg_training_length] ))
training_filenames = np.concatenate( ( pos_filenames_shuff[:pos_training_length], neg_filenames_shuff[:neg_training_length] ))

training_images, training_labels, training_filenames = shuffle( training_images, training_labels, training_filenames, random_state=0 )

print(pos_samples)
print(neg_samples)
print(training_length)
print(test_length)
print(validation_length)

with open( os.path.join( global_dirext, 'train-labels.csv' ), 'w') as f:
	writer = csv.writer(f)
	dirname = os.path.join( global_dirext, 'train/' )
	for train_i in range( training_length ):
		writer.writerow( [training_filenames[train_i], training_labels[train_i]] )
		if training_labels[train_i] == 0:
			name = os.path.join( dirname, 'noperson', training_filenames[train_i] )
		else:
			name = os.path.join( dirname, 'person', training_filenames[train_i] )
		name = os.path.join( dirname, training_filenames[train_i] )
		im = Image.fromarray( np.reshape( training_images[train_i], (120, 160) ) )
		im.save( name )

validation_images = np.vstack( ( pos_images_shuff[pos_training_length:pos_training_length+pos_validation_length],
	neg_images_shuff[neg_training_length:neg_training_length+neg_validation_length] ))
validation_labels = np.concatenate( ( pos_labels_shuff[pos_training_length:pos_training_length+pos_validation_length], neg_labels_shuff[neg_training_length:neg_training_length+neg_validation_length] ))
validation_filenames = np.concatenate( ( pos_filenames_shuff[pos_training_length:pos_training_length+pos_validation_length], neg_filenames_shuff[neg_training_length:neg_training_length+neg_validation_length] ))

validation_images, validation_labels, validation_filenames = shuffle( validation_images, validation_labels, validation_filenames, random_state=0 )

with open( os.path.join( global_dirext, 'validate-labels.csv' ), 'w') as f:
	writer = csv.writer(f)
	dirname = os.path.join( global_dirext, 'validate/' )
	for validate_i in range( validation_length ):
		writer.writerow( [validation_filenames[validate_i], validation_labels[validate_i]] )
		if validation_labels[validate_i] == 0:
			name = os.path.join( dirname, 'noperson', validation_filenames[validate_i] )
		else:
			name = os.path.join( dirname, 'person', validation_filenames[validate_i] )
		name = os.path.join( dirname, validation_filenames[validate_i] )
		im = Image.fromarray( np.reshape( validation_images[validate_i], (120, 160) ) )
		im.save( name )

test_images = np.vstack( ( pos_images_shuff[pos_training_length+pos_validation_length:], neg_images_shuff[neg_training_length+neg_validation_length:] ))
test_labels = np.concatenate( ( pos_labels_shuff[pos_training_length+pos_validation_length:], neg_labels_shuff[neg_training_length+neg_validation_length:] ))
test_filenames = np.concatenate( ( pos_filenames_shuff[pos_training_length+pos_validation_length:], neg_filenames_shuff[neg_training_length+neg_validation_length:] ))

test_images, test_labels, test_filenames = shuffle( test_images, test_labels, test_filenames, random_state=0 )

with open( os.path.join( global_dirext, 'test-labels.csv' ), 'w') as f:
	writer = csv.writer(f)
	dirname = os.path.join( global_dirext, 'test/' )
	for test_i in range( test_length ):
		writer.writerow( [test_filenames[test_i], test_labels[test_i]] )
		if test_labels[test_i] == 0:
			name = os.path.join( dirname, 'noperson', test_filenames[test_i] )
		else:
			name = os.path.join( dirname, 'person', test_filenames[test_i] )
		name = os.path.join( dirname, test_filenames[test_i] )
		im = Image.fromarray( np.reshape( test_images[test_i], (120, 160) ) )
		im.save( name )
'''
with open( os.path.join( global_dirext, 'test-valid-labels.csv' ), 'w') as f:
	writer = csv.writer(f)
	dirname = os.path.join( global_dirext, 'train-valid-combined/')
	for train_i in range( training_length ):
		writer.writerow( [training_filenames[train_i], training_labels[train_i]] )
		if training_labels[train_i] == 0:
			name = os.path.join( dirname, 'noperson', training_filenames[train_i] )
		else:
			name = os.path.join( dirname, 'person', training_filenames[train_i] )
		name = os.path.join( dirname, training_filenames[train_i] )
		im = Image.fromarray( np.reshape( training_images[train_i], (120, 160) ) )
		im.save( name )
	for validate_i in range( validation_length ):
		writer.writerow( [validation_filenames[validate_i], validation_labels[validate_i]] )
		if validation_labels[validate_i] == 0:
			name = os.path.join( dirname, 'noperson', validation_filenames[validate_i] )
		else:
			name = os.path.join( dirname, 'person', validation_filenames[validate_i] )
		name = os.path.join( dirname, validation_filenames[validate_i] )
		im = Image.fromarray( np.reshape( validation_images[validate_i], (120, 160) ) )
		im.save( name )
'''
