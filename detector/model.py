import cv2  # OpenCV library for computer vision tasks

class ObjectDetectionModel:
    def __init__(self, model_path, config_path):
        """
        Constructor to initialize model
        
        model_path -> path to frozen model (.pb file)
        config_path -> path to config (.pbtxt file)
        """

        # Load DNN model using OpenCV
        self.model = cv2.dnn_DetectionModel(model_path, config_path)

        # Set input size (image resized to 320x320)
        self.model.setInputSize(320, 320)

        # Normalize pixel values
        self.model.setInputScale(1.0 / 127.5)

        # Mean subtraction (for normalization)
        self.model.setInputMean((127.5, 127.5, 127.5))

        # Swap R and B channels (OpenCV uses BGR, model expects RGB)
        self.model.setInputSwapRB(True)

    def get_model(self):
        """
        Returns the loaded model
        """
        return self.model