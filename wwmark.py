import os
import sys
import cv2
import ffmpeg
import tempfile
from enum import Enum
from PIL import Image
from pdf2image import convert_from_path


class OverlayText(Enum):
    DEFAULT = BOTTOMRIGHT = {
        "x" : "w-tw-10",
        "y" : "h-th-10",
        "alpha" : "0.5",    
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
    DEFAULT = BOTTOMRIGHT = {
        "x": "main_w-overlay_w-5",
        "y": "main_h-overlay_h-5",
    }

class Wwmark(object):

    def __init__(self, i_file, i_mark, o_file, **kwargs):

        self.i_file = i_file if os.path.exists(i_file) else sys.exit(1)
        self.i_mark = i_mark
        self.o_file = o_file
        self.kwargs = kwargs
    
    def pdf(self,mark="text"):
        ims = []

        with tempfile.TemporaryDirectory() as temp:
    
            images = convert_from_path(self.i_file, output_folder=temp, thread_count = 1, output_file = lambda x: x+1)  # Original uuid generator is not sortable
            
            for item in sorted(os.listdir(temp)):

                backgroud = ffmpeg.input(os.path.join(temp,item))

                if mark == "text":
                    ol = ffmpeg.drawtext(backgroud, text=self.i_mark, **self.kwargs)
                
                if mark == "image":
                    ol = ffmpeg.overlay(backgroud, ffmpeg.input(self.i_mark), **self.kwargs)
                
                it = os.path.join(temp,"{}.jpg".format(item))   # if format with png, that's would be problem with RGBA, im.convert('RGB')

                self.save(ol, path=it)

                with open(it,'rb') as f:
                    im = Image.open(f)
                    im.load()
                    ims.append(im)
                    # ims.append(im.convert('RGB'))
      
        ims[0].save(self.o_file, "PDF" ,resolution=100.0, save_all=True, append_images=ims[1:])

    def text(self):
        self.save(ffmpeg.drawtext(ffmpeg.input(self.i_file), text=self.i_mark, **self.kwargs))
        
    def image(self):
        self.save(ffmpeg.overlay(ffmpeg.input(self.i_file), ffmpeg.input(self.i_mark), **self.kwargs))

    def save(self,stream, path=None):
        if path is None:
            path = self.o_file

        ol = ffmpeg.output(stream, path)
        ffmpeg.run(ol,quiet=True)