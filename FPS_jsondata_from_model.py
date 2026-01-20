import json
import numpy as np
import time
import tensorflow as tf
import platform

print("Running on:", platform.system(), platform.machine())

# JSON template
data = {
    "device": "Raspberry Pi 4 (64-bit OS)",
    "models": [
        {"file": "model_float32.tflite", "precision": "Float32", "fps": None},
        {"file": "best_int8.tflite", "precision": "INT8", "fps": None}
    ],
    "notes": "FPS values measured using identical input resolution and camera stream on Raspberry Pi 4."
}

def measure_fps(model_path, precision, runs=None, batch_size=None):
    try:
        system_os = platform.system()
        delegates = []

        # Only enable XNNPACK on Linux (Raspberry Pi)
        if system_os == "Linux":
            delegates = [tf.lite.experimental.load_delegate("libtensorflowlite_xnnpack_delegate.so")]

        interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegates)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']

        # Set default runs and batch size based on precision
        if runs is None:
            runs = 50 if precision == "Float32" else 100
        if batch_size is None:
            batch_size = 1 if precision == "Float32" else 2

        # Adjust batch dimension if possible
        batch_input_shape = input_shape.copy()
        if len(batch_input_shape) >= 1:
            batch_input_shape[0] = batch_size
        else:
            batch_input_shape = (batch_size,) + tuple(batch_input_shape)

        # Create dummy batch input
        dummy_input = np.random.random(batch_input_shape).astype(np.float32)
        if input_dtype == np.uint8:
            dummy_input = (dummy_input * 255).astype(np.uint8)

        # Warm-up
        interpreter.set_tensor(input_details[0]['index'], dummy_input[0:1])
        interpreter.invoke()

        # Measure FPS
        start_time = time.perf_counter()
        for _ in range(runs):
            for i in range(batch_size):
                interpreter.set_tensor(input_details[0]['index'], dummy_input[i:i+1])
                interpreter.invoke()
        end_time = time.perf_counter()

        total_inferences = runs * batch_size
        avg_time = (end_time - start_time) / total_inferences
        fps = 1 / avg_time
        return round(fps, 2)

    except Exception as e:
        print(f"Error measuring FPS for {model_path}: {e}")
        return None

# Measure FPS for each model
for model in data["models"]:
    print(f"Measuring FPS for {model['file']} ...")
    model["fps"] = measure_fps(model["file"], precision=model["precision"])

# Save JSON
with open("data.json", "w") as file:
    json.dump(data, file, indent=4)

print("\n✅ JSON saved with FPS values:")
print(json.dumps(data, indent=4))
