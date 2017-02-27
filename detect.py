
from utils import *
import os.path

image = mpimg.imread('test_images/test1.jpg')

templist = ['test_images/cutouts/cutout1.jpg', 'test_images/cutouts/cutout2.jpg', 'test_images/cutouts/cutout3.jpg',
            'test_images/cutouts/cutout4.jpg', 'test_images/cutouts/cutout5.jpg', 'test_images/cutouts/cutout6.jpg']


def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def find_cars(img, ystart, ystop, scale,
              spatial_size=(16, 16), hist_bins=32, orient=12,
              pix_per_cell=8, cell_per_block=2):

    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:, :, 0])
    img = img.astype(np.float32) / 255
    img_boxes = []

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Sliding window
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # extract the image patch
            subimg = cv2.resize(
                ctrans_tosearch[ytop:ytop + window, xleft:xleft + window],
                (64, 64)
            )

            # get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # scale features and make a prediction
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = model.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img,
                              (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + ystart + win_draw),
                              (0, 0, 255), 6)
                img_boxes.append((
                    (xbox_left, ytop_draw + ystart),
                    (xbox_left + win_draw, ytop_draw + ystart + win_draw)))
                heatmap[ytop_draw + ystart:ytop_draw + win_draw + ystart,
                xbox_left:xbox_left + win_draw] += 1

    return draw_img, heatmap, img_boxes


if not os.path.exists('classifier.p'):
    model, scaler = train(color_space='YCrCb', spatial_size=(16, 16), hist_bins=32, orient=12,
                            pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True,
                            hist_feat=True, hog_feat=True)
    save_model(model, scaler)
else:
    model, scaler = load_model()

# ystart = 400
# ystop = 656
# scale = 1.5
#
# images = glob.glob('test_images/*.jpg')
# img = mpimg.imread('test_images/test1.jpg')
#
# out_img, heatmap, img_boxes = find_cars(img, ystart, ystop, scale)
# heatmap = apply_threshold(heatmap, 1)
# labels = label(heatmap)
# draw_img = draw_labeled_bboxes(img, labels)

