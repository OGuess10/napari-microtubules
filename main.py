from magicgui import magicgui
import matplotlib.pyplot as plt
import numpy as np
import typing

import napari
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import Notification, NotificationSeverity

import handleImage
from handleImage import LoadTIFF


def notifyUser(message: str, severity=NotificationSeverity.WARNING):
    """ show Napari notification """
    notif = Notification(
        message,
        severity,
        actions=[("OK", lambda x: None)],
    )
    NapariQtNotification.show_notification(notif)


def preprocessVideo(image_layer):
    """ prep video data for processing """
    video = np.array(image_layer.data, dtype=np.uint16)
    return video


def processSingleFrame(img, line, struct_size, thres_ratio):
    """ process one frame using user-chosen line """
    result = handleImage.detectLine(img, line, struct_size, thres_ratio)
    if result is None:
        notif = Notification(
                f'no detected microtubule at frame. draw a line at more frames.',
                NotificationSeverity.WARNING,
                actions=[('OK', lambda x: None)],
            )
        NapariQtNotification.show_notification(notif)
        return
    return result


def visualizeSegmentation(viewer, video, lengths):
    """ add segmented overlay and plots lengths """
    if viewer:
        viewer.add_image(
            video,
            name="segmented microtubule",
            colormap="red",
            blending="additive",
        )

    # plot microtubule length over time
    plt.plot(range(len(lengths)), lengths, color="r")
    plt.title("length of microtubule over time")
    plt.xlabel("frame in video")
    plt.ylabel("length")
    plt.legend(["length"], loc="best")
    plt.show()


@magicgui(call_button="run")
def main(
        draw_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        struct_size=7,
        start_frame=0,
        end_frame=71,
        thres_ratio=1.0,
        viewer: napari.Viewer = None
) -> typing.List[napari.types.LayerDataTuple]:
    frame2line = {}
    if draw_layer is not None:
        lines = draw_layer.data
        for line in lines:
            frame2line[line[0][0]] = line
    
    video = preprocessVideo(image_layer)
    length = []
    tiff_loader = LoadTIFF(video)

    if start_frame not in frame2line:
        notifyUser('draw a line to pick a microtubule', NotificationSeverity.WARNING)
        return
    
    # process each frame
    for i in range(start_frame, min(end_frame, len(video))):
        img = tiff_loader.tiff_gray_image[i]
        if i in frame2line:
            line = frame2line[i]

        ret = processSingleFrame(img, line, struct_size, thres_ratio)
        
        end_points, thres_img, l = ret
        line = [[i, end_points[0][0], end_points[0][1]], [i, end_points[1][0], end_points[1][1]]]

        scaled_img = np.clip(thres_img.astype(np.uint32) * 257, 0, 65535).astype(np.uint16)
        video[i] = scaled_img
        length.append(l)

    visualizeSegmentation(viewer, video, length)
    metadata = {"name": "segment", "colormap": "red", "blending": "additive"}
    return [(video, metadata, "image")]


@magicgui(call_button="save to local")
def saveImageLocally(image_layer: "napari.layers.Image"):
    """ save segmented image to local file """
    image_layer.save("segmentation.tif", plugin="builtins")


def initializeNapari():
    """ initialize Napari and widgets """
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(main)
    viewer.window.add_dock_widget(saveImageLocally)
    napari.run()


if __name__ == "__main__":
    initializeNapari()
