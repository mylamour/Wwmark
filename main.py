import os
import sys
import cv2
import json
import click
import magic
import ffmpeg
import inspect

from wwmark import Wwmark, OverlayImage, OverlayText

mime = magic.Magic(mime=True)

def setting(i,p,b):

    if mime.from_file(i) == "application/pdf":
        i_type = "pdf"

    elif mime.from_file(i).startswith("video/"):
        i_type = "video"
        b = False   # video type force to False, because it can't be blind

    elif mime.from_file(i).startswith("image/"):
        i_type = "image"

    else:
        sys.exit(1)

    if p not in (OverlayText.DEFAULT.value, OverlayImage.DEFAULT.value ):
        p = json.loads(p)

    return i_type, p, b

@click.group()
def cli():
    pass

@cli.command()
@click.option('-f', help="Advanced mode, All from config file")
def config(f):
    if os.path.exists(f):
        with open(f,'r', encoding="utf-8") as config:
            c = json.load(config)
            if str.lower(c['action']) == "text":
                Wwmark(i_file=c['i'], i_mark=c['m'], o_file=c['o'], blind=c['b'], **c['p']).text()
            elif str.lower(c['action']) == "image":
                Wwmark(i_file=c['i'], i_mark=c['m'], o_file=c['o'], blind=c['b'], **c['p']).image()

@cli.command()
@click.option('-i', help="Your input file path, it can be image or video")
@click.option('-m', help="Your mark file path, it only can be text")
@click.option('-o', help="Your output file path, ")
@click.option('-p', help='mark postion, with like this { "x": "15","y": "5"}', default=OverlayText.DEFAULT.value)
@click.option('--blind/--no-blind', help='Blind or not?', default=False)
def text(i, m, o, blind, p):

    i_type, p, b = setting(i,p,blind)

    if i_type == "pdf":
        return Wwmark(i_file=i, i_mark=m, o_file=o, blind=b, **p).pdf(inspect.stack()[0][3])

    Wwmark(i_file=i, i_mark=m, o_file=o, blind=b, **p).text()

@cli.command()
@click.option('-i', help="Your input file path, it can be image or video")
@click.option('-m', help="Your mark file path, it can be image or video")
@click.option('-o', help="Your output file path, ")
@click.option('-p', help='mark postion, with like this {"x": "main_w-overlay_w-5","y": "5"}', default=OverlayImage.DEFAULT.value)
@click.option('--blind/--no-blind', help='Encode your image', default=False)
def image(i, m, o, blind, p):

    i_type, p, b = setting(i,p,blind)

    if i_type == "pdf":
        return Wwmark(i_file=i, i_mark=m, o_file=o, blind=b, **p).pdf(inspect.stack()[0][3])

    Wwmark(i_file=i, i_mark=m, o_file=o, blind=b, **p).image()


@cli.command()
@click.option('-i', help="Your original image path, text blind watermark didn't need it")
@click.option('-m', help="Your blind watermark image", default=None)
@click.option('-o', help="Your output file path")
@click.option('--type', help="Your blind file type, image or text", default="image")
def show(i, m, o, type):
    if str.lower(type) == "image" and i == None:
        # Only image with text blind, it's necessary need original image
        sys.exit(1)

    return Wwmark(i_file=i, i_mark=m, o_file=o, blind=None).show(str.lower(type))

# @click.group(chain=True)
# def cli2():
#     pass

# cli = click.CommandCollection(sources=[cli, cli2])

if __name__ == "__main__":
    cli()