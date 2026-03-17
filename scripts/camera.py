import cv2
import os
import numpy as np
import random
import threading


class Webcam:
    def __init__(self, image_test=None):
        self.image_test = image_test

        if self.image_test is None:
            # Use real camera
            self.cam = cv2.VideoCapture(0)
        else:
            # Use static image
            if not os.path.exists(self.image_test):
                raise FileNotFoundError(f"Image not found: {self.image_test}")
            self.static_image = cv2.imread(self.image_test)
            if self.static_image is None:
                raise ValueError(f"Failed to load image: {self.image_test}")

    def feed(self):
        """
        Returns a tuple: (success_flag, frame)
        """
        if self.image_test is None:
            # real camera
            ret, frame = self.cam.read()
            if not ret:
                return ret, None
            return ret, frame.copy()
        else:
            # return a copy of the static image
            return True, self.static_image.copy()

class raw_Webcam(Webcam):
    def __init__(self, image_test=None):
        """
        Raw webcam inherits from Webcam.
        Can use real webcam or a test image.
        """
        super().__init__(image_test)

class art_Webcam:

    def __init__(self, pixel=True, color=True, column=True, image_test=None):
        self.raw_cam      = raw_Webcam(image_test)
        self.h, self.w, _ = self.raw_cam.feed()[1].shape
        self.set_artifact(pixel, color, column)

    def set_artifact(self, pixel_art, color_art, column_art):
        
        self.pixel = pixel_art
        self.color = color_art
        self.column = column_art

        self.dead_column  = self.__get_dead_column()
        self.dead_pixels  = self.__get_dead_pixel()
        self.stuck_pixels = self.__get_colorimetry()

    def __get_dead_pixel(self):
        num_pixels = random.randint(1, 5)
        return [(random.randint(0, self.w - 1), random.randint(0, self.h - 1)) for _ in range(num_pixels)]

    def __get_colorimetry(self):
        num_stuck = random.randint(1,10)
        return [
            (random.randint(0, self.w - 1), random.randint(0, self.h - 1),
            [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            for _ in range(num_stuck)
        ]
    
    def __get_dead_column(self):
        n_column = random.randint(1, 5)
        return [random.randint(0, self.w) for _ in range(n_column)]
    
    def feed(self):
        ret, frame = self.raw_cam.feed()
        if self.pixel :
            for (x, y) in self.dead_pixels:
                frame[y, x] = [0, 0, 0]
        if self.color :
            for (x, y, color) in self.stuck_pixels:
                frame[y, x] = color

        if self.column :
            for col in self.dead_column:
                frame[:, col] = 0
         
        return ret, frame

class Camera:

    def __init__(self, raw=True, art=True, image_test=None):
        
        self.running = False
        self.thread = None

        if raw and art:
            print("All True")
            self.art_cam = art_Webcam(image_test=image_test)
            self.raw_cam = self.art_cam.raw_cam
        elif raw:
            print("Only raw")
            self.raw_cam = raw_Webcam(image_test=image_test)
            self.art_cam = None
        elif art:
            print("Only art")
            self.raw_cam = None
            self.art_cam = art_Webcam(image_test=image_test)
        else:
            print("No feed")
            self.raw_cam = None
            self.art_cam = None

    def set_artifact(self, pixel_art, color_art, column_art):
        if self.art_cam == None:
            print("No artifact cam to setup !")
            return 1

        self.art_cam.set_artifact(pixel_art, color_art, column_art)
        print(f"Artifact setted up for (pixel_art={pixel_art} / color_art={pixel_art} / column_art={column_art})")
        return 0

    def on(self):

        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def off(self):

        self.running = False

        if self.thread is not None:
            self.thread.join()

    def _loop(self):

        while self.running:

            self.snapshot()

            if self.raw_cam is not None:
                cv2.imshow("raw feed", self.raw_frame)

            if self.art_cam is not None:
                cv2.imshow("art feed", self.art_frame)


    def snapshot(self):
        if self.raw_cam != None:
            rval, self.raw_frame = self.raw_cam.feed()
                    
        if self.art_cam != None:
            _, self.art_frame_post = self.art_cam.feed()
            light = np.zeros_like(self.art_frame_post)
            cv2.circle(light, (300, 20), 300, (255, 255, 255), -1)
            cv2.circle(light, (600, 50), 280, (255, 255, 255), -1)
            light = cv2.GaussianBlur(light, (151, 151), 0)
            self.art_frame = cv2.addWeighted(self.art_frame_post, 0.7, light, 0.3, 0)

    def save(self, path="test/"):
        if path not in os.listdir():
            os.makedirs(path, exist_ok=True)
        if self.raw_cam!=None:
            cv2.imwrite(os.path.join(path, "raw_cam.png"), self.raw_frame)
        if self.art_cam!=None:
            cv2.imwrite(os.path.join(path, "art_cam.png"), self.art_frame_post)
            cv2.imwrite(os.path.join(path, "art_cam_blur.png"), self.art_frame)
        else:
            print("No camera")


