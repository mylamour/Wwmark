import os
import sys
import cv2
import random
import ffmpeg
import tempfile
import numpy as np
from enum import Enum
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm
from colorama import Fore 


BLIND = {
    "seed": 20200209,
    "alpha": 3.0
}

class OverlayText(Enum):
    DEFAULT = BOTTOMRIGHT = {
        "x": "w-tw-10",
        "y": "h-th-10",
        "alpha": "0.5",
        "fontsize": "16",
        "fontfile": "DroidSansFallbackFull.ttf",
    }


class OverlayImage(Enum):

    DEFAULT = TEST = {
        "x": "(main_w-overlay_w)/2",
        "y": "(main_h-overlay_h)/2",
        "aa": "0.4",        
    }

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

def Binary2String(binary):
    index = 0
    string = []
    rec = lambda x, i: x[2:8] + (rec(x[8:], i-1) if i > 1 else '') if x else ''
    fun = lambda x, i: x[i+1:8] + rec(x[8:], i-1)
    while index + 1 < len(binary):
        chartype = binary[index:].index('0')
        length = chartype*8 if chartype else 8
        string.append(chr(int(fun(binary[index:index+length], chartype), 2)))
        index += length
    return ''.join(string)

# May be need a decorator to locate it
class Wwmark(object):

    def __init__(self, i_file, i_mark, o_file, blind, **kwargs):

        self.i_file = i_file if os.path.exists(i_file) else sys.exit(1)
        self.i_mark = i_mark
        self.o_file = o_file
        self.kwargs = kwargs
        self.blind = blind
        self.action = None  # "mark" or "clean"
    
    def clean(self, type="image"):
        if type == "image":
            self.remove(type,out=self.o_file)
        
        if type == "pdf":
            self.action = "clean"
            self.pdf()
            print("[NOTICE]: Please checkout your output file {} {}".format(Fore.RED, self.o_file))

    def remove(self,type="image",out=None):
        if self.i_mark == None:
            # Try without original mark file. Lucky ?
            img = cv2.imread(self.i_file)
            alpha = 2.0
            beta = -160
            new = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            cv2.imwrite(out, new)

        else:
            # For translucency watermark
            # Code from https://stackoverflow.com/questions/32125281/removing-watermark-out-of-an-image-using-opencv/32141019
            img = cv2.imread(self.i_file)
            gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Make a copy of the grayscale image
            bg = gr.copy()

            # Apply morphological transformations
            for i in tqdm(range(5),ascii=True, ncols=199, desc="[REMOVE] {} Processed ".format(self.i_file)):
                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (2 * i + 1, 2 * i + 1))
                bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel2)
                bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel2)

            # Subtract the grayscale image from its processed copy
            dif = cv2.subtract(bg, gr)

            # Apply thresholding
            bw = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            dark = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # Extract pixels in the dark region
            darkpix = gr[np.where(dark > 0)]

            # Threshold the dark region to get the darker pixels inside it
            darkpix = cv2.threshold(darkpix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # Paste the extracted darker pixels in the watermark region
            bw[np.where(dark > 0)] = darkpix.T

            cv2.imwrite(out, bw)

    def show(self, mark="image"):
        if mark == "image":
            img = cv2.imread(self.i_file)
            img_wm = cv2.imread(self.i_mark)

            random.seed(BLIND.get("seed"))
            m, n = list(
                range(int(img.shape[0] * 0.5))), list(range(img.shape[1]))
            random.shuffle(m)
            random.shuffle(n)

            f1 = np.fft.fft2(img)
            f2 = np.fft.fft2(img_wm)

            rwm = (f2 - f1) / BLIND.get("alpha")
            rwm = np.real(rwm)

            wm = np.zeros(rwm.shape)
            for i in range(int(rwm.shape[0] * 0.5)):
                for j in range(rwm.shape[1]):
                    wm[m[i]][n[j]] = np.uint8(rwm[i][j])
            for i in tqdm(range(int(rwm.shape[0] * 0.5)), ncols=199, ascii=True, desc="[EXTRACT] {} try to find blind watermark".format(self.i_file),total=None):
                for j in range(rwm.shape[1]):
                    wm[rwm.shape[0] - i - 1][rwm.shape[1] - j - 1] = wm[i][j]
            assert cv2.imwrite(self.o_file, wm)
            print("[NOTICE]: Please checkout your output file {} {}".format(Fore.RED, self.o_file))

        else:
            # I was wondering why it didn't work
            if mark == "pdf":
                with tempfile.TemporaryDirectory() as temp:
                    convert_from_path(self.i_file, output_folder=temp,
                                    thread_count=1, output_file=lambda x: x+1)

                    self.i_file = os.path.join(temp, sorted(os.listdir(temp))[0])
                    img = Image.open(self.i_file).convert("RGBA")
            
            # Decode text blind image 
            else:
                img = Image.open(self.i_file)

            pixels = list(img.getdata())
            binary = ''.join([str(int(r >> 1 << 1 != r))+str(int(g >> 1 << 1 != g))+str(int(b >> 1 << 1 != b))+str(int(t >> 1 << 1 != t))
                              for (r, g, b, t) in tqdm(pixels, ncols=199, ascii=True, desc="[EXTRACT] {} try to find blind watermark".format(self.i_file))])
            location = binary.find('0'*16)
            endIndex = location+(8-(location%8)) if location%8!=0 else location

            data = Binary2String(binary[0: endIndex])
            print("[NOTICE]: {} {}".format(Fore.RED, data))

    def pdf(self, mark="text"):

        locs = None
    
        if self.kwargs.get("location"):
            flag = self.kwargs.pop("location")
            if "," in flag:
                locs = list(map(lambda x: int(x)-1,flag.split(",")))
            elif "-" in flag:
                loc = list(map(int,flag.split('-'))) 
                locs = list(range(loc[0]-1 ,loc[1]))
            else:
                locs = [int(flag)]

        ims = []

        with tempfile.TemporaryDirectory() as temp:

            # Original uuid generator is not sortable
            convert_from_path(self.i_file, output_folder=temp,
                              thread_count=1, output_file=lambda x: x+1)
            
            items = sorted(os.listdir(temp))

            for k, item in tqdm(enumerate(items), ncols=199, ascii=True, desc="[{}] {} watermark with {}".format(str.upper(self.action) if self.action else "ADD", self.i_file, self.i_mark), unit="page"):

                self.i_file = os.path.join(temp, item)
                it = os.path.join(temp, "{}.png".format(item))

                #  this part code look like dirty
                #  if not found special locs, just rename it
                if locs != None and locs.count(k) == 0:
                    os.rename(self.i_file,it)

                else:
                    if self.action == "clean":
                        self.remove(out=it)
                    
                    else:
                        if mark == "text":
                            self.text(path=it)

                        if mark == "image":
                            self.image(path=it)

                with open(it, 'rb') as f:
                    im = Image.open(f)
                    im.load()
                    # ims.append(im)
                    ims.append(im.convert('RGB'))

        ims[0].save(self.o_file, "PDF", resolution=100.0,
                    save_all=True, append_images=ims[1:])
        
        print("[NOTICE]: Please checkout your output file {} {}".format(Fore.RED, self.o_file))

    def text(self, path=None):
        if not self.blind:
            return self.save(ffmpeg.drawtext(ffmpeg.input(self.i_file), text=self.i_mark, **self.kwargs),path)

        # blind text based on lsb
        # This part code modified from internet

        def zero_lsb(img):
            pixels = list(img.getdata())
            pixels_new = [(r>>1<<1, g>>1<<1, b>>1<<1, t>>1<<1) for (r, g, b, t) in pixels]
            img_new = Image.new(img.mode, img.size)
            img_new.putdata(pixels_new)
            return img_new

        def binary(integer):
            return "0"*(8-(len(bin(integer))-2))+bin(integer).replace('0b','')

        img = Image.open(self.i_file).convert("RGBA")       # Convert it to 4 channel
        img_zlsb = zero_lsb(img)

        data_bin = ''.join(map(binary, bytearray(self.i_mark, 'utf-8')))

        encodedPixels = [(r+int(data_bin[index*4+0]),
                            g+int(data_bin[index*4+1]),
                            b+int(data_bin[index*4+2]),
                            t+int(data_bin[index*4+3])) if index*4 < len(data_bin) else (r,g,b,t) for index, (r,g,b,t) in tqdm(enumerate(list(img_zlsb.getdata())), ncols=199, ascii=True, desc="[ADD] {} watermark with {}".format(self.i_file, self.o_file))]
        encodedImage = Image.new(img.mode, img.size)
        encodedImage.putdata(encodedPixels)
        encodedImage.save(path if path else self.o_file)

        print("[NOTICE]: Please checkout your output file {} {}".format(Fore.RED, self.o_file))

    def image(self, path=None):
        if not self.blind:
            #  colorchannelmixer would make it transparent
            #  Add "aa" as part of config, but still pass parameters with self.kwarges
            return self.save(ffmpeg.overlay(ffmpeg.input(self.i_file), ffmpeg.input(self.i_mark).colorchannelmixer(aa=self.kwargs.pop("aa") if self.kwargs.get("aa") else "0.5" ), **self.kwargs),path)

        # Blind Image
        # This part code modified from https://github.com/chishaxie/BlindWaterMark
        img = cv2.imread(self.i_file)
        wm = cv2.imread(self.i_mark)

        h, w = img.shape[0], img.shape[1]
        hwm = np.zeros((int(h * 0.5), w, img.shape[2]))

        assert hwm.shape[0] > wm.shape[0]
        assert hwm.shape[1] > wm.shape[1]

        hwm2 = np.copy(hwm)
        for i in range(wm.shape[0]):
            for j in range(wm.shape[1]):
                hwm2[i][j] = wm[i][j]

        random.seed(BLIND.get("seed"))
        m, n = list(range(hwm.shape[0])), list(range(hwm.shape[1]))
        random.shuffle(m)
        random.shuffle(n)
        for i in range(hwm.shape[0]):
            for j in range(hwm.shape[1]):
                hwm[i][j] = hwm2[m[i]][n[j]]

        rwm = np.zeros(img.shape)
        for i in range(hwm.shape[0]):
            for j in range(hwm.shape[1]):
                rwm[i][j] = hwm[i][j]
                rwm[rwm.shape[0] - i - 1][rwm.shape[1] - j - 1] = hwm[i][j]

        f1 = np.fft.fft2(img)
        f2 = f1 + BLIND.get("alpha") * rwm

        img_wm = np.real(np.fft.ifft2(f2))

        assert cv2.imwrite(path if path else self.o_file, img_wm, [
                           int(cv2.IMWRITE_JPEG_QUALITY), 100])
                           
        print("[NOTICE]: Please checkout your output file {} {}".format(Fore.RED, self.o_file))

    def save(self, stream, path=None):
        if path is None:
            path = self.o_file

        ol = ffmpeg.output(stream, path)
        ffmpeg.run(ol, overwrite_output=True, quiet=True)