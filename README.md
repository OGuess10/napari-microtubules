# tracking microtubules
## our project
the goal of this project is to track the growth and shrinkage of a microtubule over time

- we've created a graphical user intergace (GUI) using Napari/Python that allows users to select a microtubule to track
- the user-selected microtubule is then segmented based on each frame of the corresponding video to track its growth and shrinkage
- our interface reports the growth/shrinkage events, their duration, and rate of individual microtubule growth/shrinkage via a graph


## features
- GUI interface
   - users can input video data in TIFF format
   - users can select a microtubule in a frame
   - users can scrub frame-by-frame after segmentation and reselect the microtubule in each frame if wrongly selected
   - after analysis, program creates a plot to display microtubule growth/shrinkage information
- segmentation
   - segmentation algorithm segments a user-chosen microtubule
   - manual user inputs is limited
   - algorithm effectively speeds up process of manual segmentation


## to setup on your machine:
- install the most recent version of python
- initialize a virtual environment and install dependencies:
    - navigate to the project directory
    - run `python3 -m venv venv`
    - activate the virtual environment:
        - for macOS & linux: `. venv/bin/activate`
        - for windows: `.\venv\Scripts\activate`
    - install dependencies manually, or run `pip install -r dependencies.txt`


## to run the napari plugin:
- run napari
- select Microtubules Segmentation from the Plugins menu
- click Run on the right-hand side
