## Header
- Title: Color tracker: using OpenCV for automatic color-aware region selection
- Date: 02/09/2019
- Author: Lucas dos S. Althoff
- E-mail: ls.althoff@gmail.com 

Region selection using Python and OpenCV

## Dependencies

Python 3.5.2, OpenCV 3.2.0, numpy 

## Installation
requires *conda* and numpy

install the OpenCV library

$ conda install --channel https://conda.anaconda.org/menpo opencv3
$ conda install -c anaconda numpy

Instructions
Run the code inside Lucas_Althoff folder

$ python src\main.py 

Select pixel coordinates for color region selection using left-click

press 'w' key for webcam input

press 'i' to analyse image

press 'v' to analyse video

press 'q' to stop

Other interactive inputs will be shown in terminal

Our report examples were done with:
data\mariobros.avi
data\cat.jpg
data\i76.jpg
data\i94.jpg

## Intro

The purpose of this work is to explore OpenCV package giving simple solutions for real-time video or image user interaction functionallities.

To do that we propose four specific goals 

1. Present a simple GUI that allows users to navigate in an image or video frames tracking pixels coordinate (row,column);

2. Automatically mark image/frame regions that has a color within a tolerance factor relative to the selected pixel;

3. Incorporate this interactive functionality to a real-time webcam video streaming 

## Directory tree

FirstName_LastName
├── README.md
├── FirstName_LastName.pdf
├── /relatorio
│   └── LaTeX source code 
├── /src
│   └── Implemented source code 
└── /data
    ├── image.jpg 
    └── video.avi 

