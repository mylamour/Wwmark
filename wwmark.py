import os
import sys
import cv2
import ffmpeg
from enum import Enum

class OverlayText(Enum):
    DEFAULT = {
        "x" : "10",
        "y" : "10",
        "fontsize" : "16",
        "fontfile" : "DroidSansFallbackFull.ttf",
    }

class OverlayImage(Enum):

    CENTER = {
        "x": "(main_w-overlay_w)/2",
        "y": "(main_h-overlay_h)/2",
    }

    TOPLEFT = {
        "x": "5",
        "y": "5",
    }

    TOPRIGFHT = {
        "x": "main_w-overlay_w-5",
        "y": "5",
    }

    BOTTOMLEFT = {
        "x": "5",
        "y": "main_h-overlay_h",
    }
    BOTTOMRIGHT = {
        "x": "main_w-overlay_w-5",
        "y": "main_h-overlay_h-5",
    }

class Wwmark(object):
    def __init__(self, i_file, i_mark, o_file, **kwargs):

        self.i_file = i_file if os.path.exists(i_file) else sys.exit(1)
        self.i_mark = i_mark
        self.o_file = o_file
        self.kwargs = kwargs
    
    def text(self):
        self.save(ffmpeg.drawtext(ffmpeg.input(self.i_file), text=self.i_mark, **self.kwargs))
        
    def image(self):
        self.save(ffmpeg.overlay(ffmpeg.input(self.i_file), ffmpeg.input(self.i_mark), **self.kwargs))

    def save(self,stream):
        ol = ffmpeg.output(stream, self.o_file)
        ffmpeg.run(ol)