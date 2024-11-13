from magicgui import magicgui
from qtpy.QtWidgets import QFileDialog
import datetime
import pathlib
import napari
from napari import Viewer
import numpy as np
from PIL import Image as PILImage

# Function to load frames from TIFF file
def load_tiff_frames(tiff_path):
    frames = []
    if tiff_path:
        try:
            with PILImage.open(tiff_path) as img:
                for i in range(img.n_frames):
                    img.seek(i)
                    frames.append(np.array(img))
        except Exception as e:
            print(f"Error loading TIFF file: {e}")
    return frames

# Main widget with dropdown, slider, and navigation buttons
@magicgui(
    call_button="Load TIFF Video",
    slider={"widget_type": "Slider", "min": 0, "max": 100, "step": 1},
    layout="vertical"
)
def video_widget(viewer: Viewer, slider=0):
    # Open file dialog
    file_path, _ = QFileDialog.getOpenFileName(
        caption="Select a .tiff video file",
        filter="TIFF files (*.tiff *.tif)"
    )
    
    if not file_path:
        print("No file selected.")
        return
    
    # Load frames from TIFF file
    frames = load_tiff_frames(file_path)
    if not frames:
        print("No frames found or failed to load the TIFF file.")
        return
    
    # Set slider max to match the number of frames
    video_widget.slider.max = len(frames) - 1  # Update slider max based on frame count
    
    # Display the first frame
    layer = viewer.add_image(frames[0], name="TIFF Frame", colormap="gray")

    # Function to update the displayed frame
    def update_frame(index):
        layer.data = frames[index]
    
    # Connect the slider to update the frame on change
    @video_widget.slider.changed.connect
    def on_slider_change(event):
        update_frame(video_widget.slider.value)
    
    # Add Next and Previous Frame buttons
    @magicgui(call_button="Next Frame")
    def next_frame():
        if video_widget.slider.value < len(frames) - 1:
            video_widget.slider.value += 1  # Move to the next frame
            update_frame(video_widget.slider.value)

    @magicgui(call_button="Previous Frame")
    def previous_frame():
        if video_widget.slider.value > 0:
            video_widget.slider.value -= 1  # Move to the previous frame
            update_frame(video_widget.slider.value)

    # Add buttons and widget to the viewer dock
    viewer.window.add_dock_widget(next_frame, area="bottom")
    viewer.window.add_dock_widget(previous_frame, area="bottom")

def run():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(video_widget, area="bottom")
