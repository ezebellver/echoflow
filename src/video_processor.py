import cv2
import numpy as np
from image_processor import ImageProcessor


class VideoProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
        self.prev_edges = None

    def edge_subtraction(self, current_edges, stability_threshold=20):
        """
        Perform edge subtraction by subtracting stable edges from the current frame.

        Parameters:
        - current_edges (numpy.ndarray): Binary image representing the current edges.
        - stability_threshold (int): Number of consecutive frames for an edge to be considered stable.

        Returns:
        - filtered_edges (numpy.ndarray): Filtered edges after subtraction.
        """

        # If previous edges are not available, initialize with current edges
        if self.prev_edges is None:
            self.prev_edges = current_edges
            return current_edges

        # Find the stable edges by comparing with previous edges
        stable_edges = np.logical_and(self.prev_edges, current_edges)

        # Update previous edges with current edges
        self.prev_edges = current_edges

        # Apply stability threshold
        stability_count = np.sum(stable_edges)
        if stability_count >= stability_threshold:
            # filtered_edges = current_edges ^ stable_edges  # Subtract stable edges from current edges
            filtered_edges = np.uint8(np.bitwise_and(current_edges, np.bitwise_not(stable_edges)) * 255)
        else:
            filtered_edges = current_edges

        # Create a binary image highlighting stable edges
        stable_edges_image = np.zeros_like(current_edges, dtype=np.uint8)
        stable_edges_image[stable_edges] = 255

        # Display the stable edges image
        cv2.imshow('Stable Edges', stable_edges_image)
        cv2.waitKey(1)  # Add a small delay to display the image

        return filtered_edges

    def process_video_stream(self, video_stream):
        """
        Preprocess input video stream by applying edge subtraction over time.

        Parameters:
        - video_stream (cv2.VideoCapture): Video stream object.
        """

        while True:
            # Read frame from the video stream
            ret, frame = video_stream.read()
            if not ret:
                break

            # Preprocess the frame
            preprocessed_frame = self.process_image(frame)

            # Perform edge detection
            current_edges = self.edge_detection(preprocessed_frame)

            # Perform edge subtraction
            filtered_edges = self.edge_subtraction(current_edges)

            # You can perform additional processing or save the filtered_edges if needed

        # Release the video stream
        video_stream.release()
