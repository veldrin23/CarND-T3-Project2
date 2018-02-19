import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import math
import scipy
import cv2



class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

        
# from my behavioural cloning project, so so long ago
def flip_image(img, flip_direction = 0):
    """
    Function to flip image
    :param img: image to flip
    :return: flipped image
    """
    return cv2.flip(img, flipCode=flip_direction)

# from my behavioural cloning project
def change_brightness(img):
    """
    Change brightness
    :param img: image array
    :return: brightened image
    """
    change_pct = random.uniform(0.4, 1.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * change_pct
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img   

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        
        background_color = np.array([255, 0, 0])
        other_road = np.array([0,0,0])
        current_road = np.array([255,0,255])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                # augment data
                # randomly choose a flip direction
                flip_direction = np.random.choice([-9,-1,0,1])
                
                # no flip
                if flip_direction == -9:
                    image = image
                    gt_image = gt_image
                # -1: flip x and y axis
                # 0: flip around x-axis
                # 1: flip around y-axis
                else: 
                    image = flip_image(image, flip_direction)
                    gt_image = flip_image(gt_image, flip_direction)    

                image = change_brightness(image)


                # 3 layers 
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_or = np.all(gt_image == other_road, axis=2)
                gt_cr = np.all(gt_image == current_road, axis=2)
                
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_or = gt_or.reshape(*gt_or.shape, 1)
                gt_cr = gt_cr.reshape(*gt_cr.shape, 1)

                gt_image = np.concatenate((gt_bg, gt_or, gt_cr), axis = 2)
                
                    
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        
        # other road
        im_softmax_ = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax_ > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        
        
        # current road
        im_softmax_ = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax_ > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[30, 144, 255, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        #street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
                


        yield os.path.basename(image_file), np.array(street_im)
        
def gen_test_output_video(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    print("something")
    for image_file in glob(os.path.join("video_images", "*.png")):
        print(image_file)
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        
        # other road
        im_softmax_ = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax_ > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        
        
        # current road
        im_softmax_ = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax_ > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[30, 144, 255, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        #street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
                


        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        
        
def save_inference_samples_video(runs_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    #output_dir = os.path.join(runs_dir, str(time.time()))
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    #os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    #print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output_video(
        sess, logits, keep_prob, input_image, os.path.join('video_images'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join("video_images_output", name), image)