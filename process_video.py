from collections import deque
from scipy.ndimage.measurements import label
import cv2
import numpy as np
from detect import find_cars, apply_threshold, draw_labeled_bboxes
from lane_detection import LaneDetector


class VideoProcessor():

    def __init__(self):
        self.frame_number = 0
        self.heatmaps = deque(maxlen=5)
        self.ld = LaneDetector()

    def process_frame(self, img):

        out_img, heatmap, _ = find_cars(img, 400, 656, 1.5)
        self.heatmaps.append(heatmap)

        if len(self.heatmaps) < 5:
            return img

        heatmaps = np.sum(np.array(self.heatmaps), axis=0).astype(np.float32)
        heatmaps = apply_threshold(heatmaps, 3)
        labels = label(heatmaps)

        # Add the lane detection stack
        img = self.ld.process_frame(img)

        # Draw the boxes
        draw_img = draw_labeled_bboxes(img, labels)

        cv2.imshow('frame', draw_img)
        cv2.waitKey(1)
        return draw_img


video_processor = VideoProcessor()


from moviepy.editor import VideoFileClip
clip = VideoFileClip("./project_video.mp4")
output_video = "./project_video_out.mp4"
output_clip = clip.fl_image(video_processor.process_frame)
output_clip.write_videofile(output_video, audio=False)