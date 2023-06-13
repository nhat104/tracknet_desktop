import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QFileDialog,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtGui
from TrackNet import TrackNet
from argparse import Namespace
import torch
from utils.inference import inference


opt = Namespace(
    **{
        "grayscale": False,
        "sequence_length": 1,
        "dropout": 0,
        "one_output_frame": False,
    }
)

model = TrackNet(opt)


class VideoPlayerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 400, 300)

        # Create a button to open a video file
        open_box = QHBoxLayout()
        open_button = QPushButton("Open Video", self)
        open_button.clicked.connect(self.open_video)
        open_box.addWidget(open_button, 0)
        open_box.addStretch()

        # Create a label to display the video frames
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Create a layout and set the label and button as its widgets
        main_layout = QVBoxLayout()
        main_layout.addLayout(open_box, 0)
        main_layout.addWidget(self.video_label)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Create a timer to update the video frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Initialize the video capture
        self.video_capture = None

    def open_video(self):
        # Open file dialog to select a video file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )

        if file_path:
            # Release any existing video capture
            if self.video_capture is not None:
                self.video_capture.release()

            # Open the new video file
            self.video_capture = cv2.VideoCapture(file_path)

            # Start the timer
            self.timer.start(33)  # Update every 33 milliseconds (30 frames per second)

    def update_frame(self):
        # Read the next frame from the video capture
        ret, frame = self.video_capture.read()

        if ret:
            # Convert the frame to RGB format
            detect_frame = inference(model, frame)
            rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a QImage from the RGB frame
            image = QImage(
                rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888
            )

            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(image)

            # Scale the pixmap to fit the label
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)

            # Set the pixmap as the label's pixmap
            self.video_label.setPixmap(scaled_pixmap)
        else:
            # Video playback has ended, stop the timer
            self.timer.stop()

            # Release the video capture
            self.video_capture.release()

            # Display a message when the video ends
            self.video_label.setText("Video playback ended.")

            # Resize the label to fit the text
            self.video_label.adjustSize()


if __name__ == "__main__":
    model.load_state_dict(
        torch.load("./utils/best-custom-dataset.pth", map_location=torch.device("cpu"))
    )
    model.to("cpu")
    model.eval()

    app = QApplication(sys.argv)
    video_player_window = VideoPlayerWindow()
    video_player_window.show()
    sys.exit(app.exec_())
