# Mask R-CNN Object Detection with OpenCV

This repository contains a Python script for real-time object detection using the Mask R-CNN (Region-based Convolutional Neural Networks) model with OpenCV. The Mask R-CNN model is capable of detecting objects and segmenting them into pixel-level masks.

## Prerequisites

Before running the code, ensure you have the following installed:

1. Python (3.6 or higher)
2. OpenCV (Open Source Computer Vision Library)
3. NumPy
4. Matplotlib

You can install the required libraries using the following command:

```bash
pip install opencv-python numpy matplotlib
```

## Getting Started

1. Clone the repository or download the `mask_rcnn_object_detection.py` file.

2. Download the Mask R-CNN model files from the TensorFlow Model Zoo:
   - [Download the .pbtxt file](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) (model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt)
   - [Download the .pb file](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) (model/frozen_inference_graph.pb)

3. Download the COCO labels file:
   - [Download the ms coco labels file](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt) (model/mscoco_labels.names)

4. Place the downloaded files in the "model" directory of this repository.

## Usage

Run the `mask_rcnn_object_detection.py` script to perform real-time object detection using your webcam:

```bash
python mask_rcnn_object_detection.py
```

The script will open a window displaying the webcam feed with detected objects and their corresponding masks overlaid on the video stream.

## Configuration

You can adjust the following parameters to fine-tune the object detection process:

- `confThreshold`: Confidence threshold for object detection (default: 0.5).
- `maskThreshold`: Threshold for the mask segmentation (default: 0.3).

## License

This code is provided under the MIT License. You are free to use, modify, and distribute it according to the terms of the [MIT License](LICENSE).

## Acknowledgments

- The Mask R-CNN implementation in this code is based on the work of the TensorFlow Object Detection API.
- The COCO labels and model files are obtained from the TensorFlow Model Zoo.

## Note

Keep in mind that real-time object detection can be resource-intensive, and the performance may vary based on your hardware capabilities. If you experience low FPS (frames per second) or other performance issues, consider using a more powerful GPU or optimizing the code for your specific use case.