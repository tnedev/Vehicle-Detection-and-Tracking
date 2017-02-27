import glob
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle


def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     vis=False, feature_vec=False):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32, bins_range=(0, 2)):
    """
    Get the color histogram feature vector
    :param img:
    :param nbins:
    :param bins_range:
    :return:
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_hog_features(cell_per_block, feature_image, hog_channel, orient, pix_per_cell):
    """
    Het the HOG feature vector
    :param cell_per_block:
    :param feature_image:
    :param hog_channel:
    :param orient:
    :param pix_per_cell:
    :return:
    """
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    return hog_features


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def convert_color(img, conv='RGB2YCrCb'):
    """
    Convert the color of an image
    :param img:
    :param conv:
    :return:
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    img_features = []
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)

    return np.concatenate(img_features)


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract the features of multiple images
    :param imgs:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """
    features = []
    for file in imgs:
        image = mpimg.imread(file)
        img_features = single_img_features(image, color_space, spatial_size,
                                           hist_bins, orient,
                                           pix_per_cell, cell_per_block, hog_channel,
                                           spatial_feat, hist_feat, hog_feat, vis=False)

        features.append(img_features)
    return features


def train(color_space='YCrCb', spatial_size=(32, 32), hist_bins=32, orient=9,
                pix_per_cell=8, cell_per_block=2, hog_channel=0,
                spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Train the algorithm with car and notcar images
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """
    notcars = glob.glob('test_images/non-vehicles/*/*.png')
    cars = glob.glob('test_images/vehicles/*/*.png')

    # Balance the image data
    sample_size = 8500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    # Get the car features vectors
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    # Get the notcar feature vectors
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Normalize features
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data randomly for training and testing
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

    # Train with SVC
    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t_training = round(time.time() - t, 2)

    # computes the accuracy
    accuracy = round(svc.score(X_test, y_test), 4)

    # Print back the training info
    print('color:{}'.format(color_space), 'spatial_size:', spatial_size,
          'orient:{}, hog:{}, hist:{}, feat:{}, time: {}s, acc: {}'.format(
              orient, hog_channel, hist_bins,
              len(X_train[0]),
              t_training, accuracy))

    return svc, X_scaler


def save_model(model, scaler):
    """
    Save the training model
    :param model:
    :param scaler:
    :return:
    """
    with open('classifier.p', 'wb') as fp:
        data = {
            'model': model,
            'scaler': scaler
        }
        pickle.dump(data, fp)


def load_model():
    """
    Load trained model
    :return:
    """
    with open('classifier.p', 'rb') as fp:
        data = pickle.load(fp)
    return data['model'], data['scaler']


def draw_labeled_bboxes(image, labels):
    """
    Draw the labeled boxes
    :param image:
    :param labels:
    :return:
    """
    img = np.copy(image)
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    return img