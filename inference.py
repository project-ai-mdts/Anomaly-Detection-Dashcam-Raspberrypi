import numpy as np
import tflite_runtime.interpreter as tflite

class PotholeDetector:
    def __init__(self, model_path, threads):
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

    def detect(self, frame):
        input_img = frame.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        self.interpreter.set_tensor(self.input_index, input_img)
        self.interpreter.invoke()

        return self.interpreter.get_tensor(self.output_index)[0]
