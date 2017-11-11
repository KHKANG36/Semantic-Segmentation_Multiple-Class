import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
import cv2
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


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
        image_paths = glob(os.path.join(data_folder, 'image', '*.png'))
        label_paths = {
            re.sub(r'gtFine_color', 'leftImg8bit', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image', '*.png'))}
        
        road_color = np.array([128, 64, 128, 255])
        car_color = np.array([0, 0, 142, 255])
        person_color = np.array([220, 20, 60, 255])
        trafficL_color = np.array([250, 170, 30, 255])
        

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                
                #file_name = os.path.basename(image_file)
                image = scipy.misc.imread(image_file)
                gt_image_file = label_paths[os.path.basename(image_file)]
                gt_image = scipy.misc.imread(gt_image_file)
                #gt_image_file = scipy.misc.imread(os.path.join(data_folder, 'gt_image', file_name))
                
                #Image Augmentation
                #rot_value=np.random.uniform(-15,15) #Rotation Value
                #siz_value=np.random.uniform(0.95,1.05) #Size Value
    
                #rows = 2048
                #cols = 1024
                #center=(cols/2,rows/2)
                #rot=cv2.getRotationMatrix2D(center,rot_value,siz_value)
                # Rotation & Size
                #image_aug=cv2.warpAffine(image,rot,(cols,rows))
                #gt_image_aug=cv2.warpAffine(gt_image,rot,(cols,rows))
                #Brightness
                #image_aug = cv2.cvtColor(image_aug,cv2.COLOR_RGB2HSV)
                #random_bright = .3+np.random.uniform()
                #image_aug[:,:,2] = image_aug[:,:,2]*random_bright
                #image_aug = cv2.cvtColor(image_aug, cv2.COLOR_HSV2RGB)
                
                image = scipy.misc.imresize(image, image_shape)
                gt_image = scipy.misc.imresize(gt_image, image_shape)
                
                #image_aug = scipy.misc.imresize(image_aug, image_shape)
                #gt_image_aug = scipy.misc.imresize(gt_image_aug, image_shape)
                
                gt_road = np.all(gt_image == road_color, axis=2)
                gt_road = gt_road.reshape(*gt_road.shape, 1) 
                gt_car = np.all(gt_image == car_color, axis=2)
                gt_car = gt_car.reshape(*gt_car.shape, 1) 
                gt_person = np.all(gt_image == person_color, axis=2) 
                gt_person = gt_person.reshape(*gt_person.shape, 1) 
                gt_trafficL = np.all(gt_image == trafficL_color, axis=2) 
                gt_trafficL = gt_trafficL.reshape(*gt_trafficL.shape, 1) 
                gt_image = np.concatenate((gt_road,gt_car,gt_person,gt_trafficL), axis=2)
                gt_bg = np.all(gt_image == 0, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1) 
                gt_image = np.concatenate((gt_image,gt_bg), axis=2)

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
    for image_file in glob(os.path.join(data_folder, 'image', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        
        #Road
        im_softmax_r = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
        segmentation_r = (im_softmax_r > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation_r, np.array([[128, 64, 128, 128]]))
        
        #Car
        im_softmax_c = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation_c = (im_softmax_c > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = mask + np.dot(segmentation_c, np.array([[0, 0, 142, 128]]))
        
        #Person
        im_softmax_p = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
        segmentation_p = (im_softmax_p > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = mask + np.dot(segmentation_p, np.array([[220, 20, 60, 128]]))
        
        #Traffic Light
        im_softmax_t = im_softmax[0][:, 3].reshape(image_shape[0], image_shape[1])
        segmentation_t = (im_softmax_t > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = mask + np.dot(segmentation_t, np.array([[250, 170, 30, 128]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
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
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def gen_video_output(sess, logits, keep_prob, image_pl, video_file, image_shape):

    cap = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    filename = 'sementic_output.mp4'
    out = cv2.VideoWriter(filename, fourcc, 29.0, (512,256))

    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        image = scipy.misc.imresize(frame, image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        
        #Road
        im_softmax_r = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
        segmentation_r = (im_softmax_r > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation_r, np.array([[128, 64, 128, 200]]))
        
        #Car
        im_softmax_c = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation_c = (im_softmax_c > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = mask + np.dot(segmentation_c, np.array([[220, 20, 60, 200]]))
        
        #Person
        im_softmax_p = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
        segmentation_p = (im_softmax_p > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = mask + np.dot(segmentation_p, np.array([[0, 0, 142, 200]]))
        
        #Traffic Light
        im_softmax_t = im_softmax[0][:, 3].reshape(image_shape[0], image_shape[1])
        segmentation_t = (im_softmax_t > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = mask + np.dot(segmentation_t, np.array([[250, 170, 30, 200]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        
        output = np.array(street_im)

        out.write(output)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
       