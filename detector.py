import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

class PotholeDetector:
    def __init__(self, model_path, threads=4):
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'] # [1, H, W, 3]

    def preprocess(self, frame):
        h, w = self.input_shape[1], self.input_shape[2]
        img = cv2.resize(frame, (w, h))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def detect(self, frame):
        inp = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])
        return preds