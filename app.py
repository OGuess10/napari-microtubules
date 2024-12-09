from magicgui import magicgui
import matplotlib.pyplot as plt
import numpy as np
import typing
from qtpy.QtWidgets import QMessageBox

import napari
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import Notification, NotificationSeverity
import napari_microtubules.handleImage as handleImage
from napari_microtubules.handleImage import LoadTIFF


segment_lengths_global = []

def showTutorial(viewer):
    msg = QMessageBox(viewer.window.qt_viewer)
    msg.setIcon(QMessageBox.Information)
    msg.setText("How to use Napari Microtubule Segmentation Plugin:")
    msg.setInformativeText(
        "Step 1: Load your data. This should be in TIFF format.\n\n"
        "Step 2: Move the slider to start on desired frame. Then add a new Shape layer and draw a line over the microtubule "
        "you would like to segment.\n\n"
        "Step 3: Ensure that the line and video layer are currently selected in the 'run segmentation' tab. "
        "Adjust the frame start and frame end numbers to match the desired length. The start frame should match the frame with the line.\n\n"
        "Step 4: Run the segmentation\n\n"
        "Step 5: If the tracking gets off, scroll to the frame you want to adjust. Then add a new shape layer "
        "and redraw a line that overlaps the microtubule. Make sure this new shape layer is selected in the "
        "'reselect microtubule' tab. Then rerun, and a new 'segmented microtubule' layer should appear with the changes.\n\n"
        "Step 5: If you would like to save the segmentation, select the segmentation layer and click the save "
        "button on the save tab.\n\n"
        "See documentation for further help: https://github.com/OGuess10/napari-microtubules"
    )
    msg.setWindowTitle("How to Use")
    msg.exec_()


def showNotification(message: str, severity=NotificationSeverity.WARNING):
    """ show Napari notification """
    notification = Notification(message, severity, actions=[("OK ", lambda x: None)])
    NapariQtNotification.show_notification(notification)


def prepareVideoData(image_layer):
    """ prep video data for processing """
    return np.array(image_layer.data, dtype=np.uint16)


def processFrame(image, line_coordinates, structuring_element_size, threshold_ratio):
    """ process a single frame using the user-chosen line """
    result = handleImage.track_microtubule(image, line_coordinates, structuring_element_size, threshold_ratio)
    if result is None:
        showNotification('cannot detect microtubule at this frame', NotificationSeverity.WARNING)
        return None
    return result


def displaySegmentationResults(viewer, processed_video, segment_lengths):
    """ add segmented overlay and plot lengths """
    if viewer:
        viewer.add_image(
            processed_video,
            name="segmented microtubule",
            colormap="red",
            blending="additive",
        )

    # make plot to show microtubule length over time
    plt.plot(range(len(segment_lengths)), segment_lengths, color="r")
    plt.title("length of microtubule over time")
    plt.xlabel("frame in video")
    plt.ylabel("length")
    plt.legend(["length"], loc="best")
    plt.show()


@magicgui(call_button="run segmentation")
def processMicrotubuleData(
        shape_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        video_start=0,
        video_end=71,
        structure=7,
        threshold=0.5,
        viewer: napari.Viewer = None
) -> typing.List[napari.types.LayerDataTuple]:
    
    frame_to_line_coordinates = {}
    if shape_layer is not None:
        drawn_lines = shape_layer.data
        for line in drawn_lines:
            frame_to_line_coordinates[line[0][0]] = line
    
    video_data = prepareVideoData(image_layer)
    segment_lengths = []
    tiff_loader = LoadTIFF(video_data)

    if video_start not in frame_to_line_coordinates:
        showNotification('draw a line to pick a microtubule', NotificationSeverity.WARNING)
        return []
    
    # process each frame
    for frame_index in range(video_start, min(video_end, len(video_data))):
        image_frame = tiff_loader.tiff_gray_image[frame_index]
        if frame_index in frame_to_line_coordinates:
            line_coordinates = frame_to_line_coordinates[frame_index]

        process_result = processFrame(image_frame, line_coordinates, structure, threshold)
        
        if process_result:
            endpoints, threshold_image, segment_length = process_result
            line_coordinates = [[frame_index, endpoints[0][0], endpoints[0][1]], [frame_index, endpoints[1][0], endpoints[1][1]]]

            scaled_image = np.clip(threshold_image.astype(np.uint32) * 257, 0, 65535).astype(np.uint16)
            video_data[frame_index] = scaled_image
            segment_lengths.append(segment_length)

    displaySegmentationResults(viewer, video_data, segment_lengths)
    metadata = {"name": "segmented microtubule", "colormap": "red", "blending": "additive"}
    return [(video_data, metadata, "image")]


@magicgui(call_button="reselect microtubule")
def reselectMicrotubule(
    shape_layer: 'napari.layers.Shapes',
    image_layer: 'napari.layers.Image',
    replace_frames_start=0,
    replace_frames_end=71,
    structure=7,
    threshold=0.5,
    hough_threshold=35,
    viewer: napari.Viewer = None
) -> typing.List[napari.types.LayerDataTuple]:
    
    # Step 1: Reset previous selection of the microtubule (clear coordinates)
    frame_to_line_coordinates = {}
    
    # Step 2: Ensure that shape_layer exists and has data to work with
    if shape_layer is not None:
        drawn_lines = shape_layer.data  # Get the drawn lines from the shape layer
        for line in drawn_lines:
            # Update the frame-to-line coordinates map with new lines
            frame_to_line_coordinates[line[0][0]] = line
    
    # Step 3: Prepare video data from image layer
    video_data = prepareVideoData(image_layer)
    segment_lengths = []
    tiff_loader = LoadTIFF(video_data)

    # Step 4: Process each frame using the new line coordinates
    for frame_index in range(replace_frames_start, min(replace_frames_end, len(video_data))):
        image_frame = tiff_loader.tiff_gray_image[frame_index]
        
        # Check if a line exists for the current frame
        if frame_index in frame_to_line_coordinates:
            line_coordinates = frame_to_line_coordinates[frame_index]

        # Process the frame with the new line coordinates
        process_result = processFrame(image_frame, line_coordinates, structure, threshold, hough_threshold)
        
        if process_result:
            endpoints, threshold_image, segment_length = process_result
            line_coordinates = [[frame_index, endpoints[0][0], endpoints[0][1]], [frame_index, endpoints[1][0], endpoints[1][1]]]

            # Update video data with processed image
            scaled_image = np.clip(threshold_image.astype(np.uint32) * 257, 0, 65535).astype(np.uint16)
            video_data[frame_index] = scaled_image
            segment_lengths.append(segment_length)

    global segment_lengths_global
    new_segment_lengths = segment_lengths_global

    # redo segmented microtubule images
    layer_name = "segmented microtubule"
    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]
        layer_data = np.array(layer.data)

        for frame_index in range(replace_frames_start, min(replace_frames_end, len(video_data))):
            if 0 <= frame_index < len(layer_data):
                layer_data[frame_index] = video_data[frame_index]
            if frame_index < len(new_segment_lengths):
                new_segment_lengths[frame_index] = segment_lengths[frame_index-replace_frames_start]

    displaySegmentationResults(viewer, layer_data, new_segment_lengths)
    metadata = {"name": "segmented microtubule", "colormap": "red", "blending": "additive"}
    return [(video_data, metadata, "image")]


@magicgui(call_button="merge segmentations")
def mergeSegmentations(
    segmentation_1: 'napari.layers.Image',
    segmentation_2: 'napari.layers.Image',
    image_layer: 'napari.layers.Image',
    video_start=0,
    video_end=71,
    viewer: napari.Viewer = None
) -> typing.List[napari.types.LayerDataTuple]:
    
    video_data = prepareVideoData(image_layer)
 
    layer_data_1 = np.array(segmentation_1.data)
    layer_data_2 = np.array(segmentation_2.data)
    new_layer_data = np.zeros_like(layer_data_1)

    for frame_index in range(video_start, min(video_end, len(video_data))):
        new_layer_data[frame_index] = layer_data_1[frame_index] + layer_data_2[frame_index]

    global segment_lengths_global

    displaySegmentationResults(viewer, new_layer_data, segment_lengths_global)
    metadata = {"name": "segmented microtubule", "colormap": "red", "blending": "additive"}
    return [(video_data, metadata, "image")]


@magicgui(call_button="save")
def saveProcessedImageLocally(image_layer: "napari.layers.Image"):
    """ save segmented image to local file """
    image_layer.save("segmentation.tif", plugin="builtins")


def initializeNapariUI():
    """ initialize viewer and widgets """
    viewer = napari.Viewer()
    showTutorial(viewer)
    viewer.window.add_dock_widget(processMicrotubuleData)
    viewer.window.add_dock_widget(reselectMicrotubule)
    viewer.window.add_dock_widget(mergeSegmentations)
    viewer.window.add_dock_widget(saveProcessedImageLocally)
    napari.run()


if __name__ == "__main__":
    initializeNapariUI()
