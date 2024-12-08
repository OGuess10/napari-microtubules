from magicgui import magicgui
import matplotlib.pyplot as plt
import numpy as np
import napari
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari_microtubules.handleTubuleSelection import *
from napari.utils.notifications import Notification, NotificationSeverity
import typing

import napari_microtubules.handleImage as handleImage
from napari_microtubules.handleImage import LoadTIFF


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
        #structure=7,
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

        process_result = processFrame(image_frame, line_coordinates, 7, threshold)
        
        if process_result:
            endpoints, threshold_image, segment_length = process_result
            line_coordinates = [[frame_index, endpoints[0][0], endpoints[0][1]], [frame_index, endpoints[1][0], endpoints[1][1]]]

            scaled_image = np.clip(threshold_image.astype(np.uint32) * 257, 0, 65535).astype(np.uint16)
            video_data[frame_index] = scaled_image
            segment_lengths.append(segment_length)

    displaySegmentationResults(viewer, video_data, segment_lengths)
    metadata = {"name": "segmented microtubule", "colormap": "red", "blending": "additive"}
    return [(video_data, metadata, "image")]


@magicgui(call_button="save")
def saveProcessedImageLocally(image_layer: "napari.layers.Image"):
    """ save segmented image to local file """
    image_layer.save("segmentation.tif", plugin="builtins")


def initializeNapariUI():
    """ initialize viewer and widgets """
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(processMicrotubuleData)
    viewer.window.add_dock_widget(saveProcessedImageLocally)
    napari.run()


if __name__ == "__main__":
    initializeNapariUI()
