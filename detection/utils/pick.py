import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
import cv2 
import sys 
def get_click_point(img):
    # Load the image

    # Display the image
    plt.imshow(img)
    plt.title("Click to select a point")
    plt.axis('on')  # Turn on axis

    # Variable to store the coordinates
    coords = []

    # Function to handle clicks and store coordinates
    def onclick(event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            coords.append((x, y))
        # Disconnect the click event and close the window
        plt.gcf().canvas.mpl_disconnect(cid)
        plt.close()

    # Connect the click event to the onclick function
    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    # Show the image and wait for clicks
    plt.show()

    # Return the coordinates
    if coords:
        print(coords[0])
        return coords[0]  # Return the first point (since we expect only one click)
    else:
        return None

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

def get_bounding_box(img):
    # Variable to store the bounding box coordinates
    bbox = None

    # Callback function to handle rectangle selection
    def onselect(eclick, erelease):
        nonlocal bbox
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        bbox = (int(x1), int(y1), int(x2), int(y2))
        plt.close()  # Close the image after selection

    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("Drag to select the bounding box")
    plt.axis('on')  # Turn on axis

    # Create a RectangleSelector
    rect_selector = RectangleSelector(ax, onselect,
                                      interactive=True,
                                      button=[1],  # Only left mouse button
                                      minspanx=5, minspany=5,
                                      spancoords='pixels')

    # Show the image and wait for the user to select the bounding box
    plt.show()

    # If a bounding box was selected, crop the image and return it
    if bbox:
        x1, y1, x2, y2 = bbox
        cropped_img = img[y1:y2, x1:x2]
        # plt.imshow(cropped_img)
        # plt.title("Cropped Image")
        # plt.axis('on')
        # plt.show()
        return cropped_img, [x1, y1, x2, y2] 
    else:
        print("No bounding box selected.")
        return None


def get_tracker(tracker_type:str = 'KCF'):
    """
    tracker_type : ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    """
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    tracker = tracker_selector(tracker_type, minor_ver)
    return tracker

def tracker_selector(tracker_type, minor_ver):
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create() 

    return tracker
    
    