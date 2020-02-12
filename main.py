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
        b = False   # look like can't save image to pdf with RGBA mode?

    elif mime.from_file(i).startswith("video/"):
        i_type = "video"
        b = False   # video type force to False

    elif mime.from_file(i).startswith("image/"):
        i_type = "image"

    else:
        sys.exit(1)

    if p:
        position = str.upper(p)

        if position in OverlayText.__members__.keys() or position in OverlayImage.__members__.keys():
            p = OverlayImage.__members__.get(position).value if OverlayImage.__members__.get(position) else OverlayText.__members__.get(position).value
        else:
            p = json.loads(p)

    return i_type, p, b

@click.group()
def cli():
    pass

@cli.command(help="Advanced mode, add watermark with config file")
@click.option('-f', help="Advanced mode, All from config file")
def config(f):
    if os.path.exists(f):
        with open(f,'r', encoding="utf-8") as config:
            c = json.load(config)
            if str.lower(c['action']) == "text":
                Wwmark(i_file=c['i'], i_mark=c['m'], o_file=c['o'], blind=c['b'], **c['p']).text()
            elif str.lower(c['action']) == "image":
                Wwmark(i_file=c['i'], i_mark=c['m'], o_file=c['o'], blind=c['b'], **c['p']).image()

@cli.command(help="Add text watermark")
@click.option('-i', help="Your input file path, it can be image or video")
@click.option('-m', help="Your mark file path, it only can be text")
@click.option('-o', help="Your output file path, ")
@click.option('-p', help='mark position, with like this { "x": "15","y": "5"}', default="DEFAULT")
@click.option('--blind/--no-blind', help='Blind or not?', default=False)
def text(i, m, o, blind, p):

    i_type, p, b = setting(i,p,blind)

    if i_type == "pdf":
        return Wwmark(i_file=i, i_mark=m, o_file=o, blind=b, **p).pdf(inspect.stack()[0][3])

    Wwmark(i_file=i, i_mark=m, o_file=o, blind=b, **p).text()

@cli.command(help="Add image watermark")
@click.option('-i', help="Your input file path, it can be image or video")
@click.option('-m', help="Your mark file path, it can be image or video")
@click.option('-o', help="Your output file path, ")
@click.option('-p', help='mark position, with like this {"x": "main_w-overlay_w-5","y": "5"},default support CENTER, TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT', default="BOTTOMRIGHT")
@click.option('--blind/--no-blind', help='Encode your image', default=False)
def image(i, m, o, blind, p):

    i_type, p, b = setting(i,p,blind)

    if i_type == "pdf":
        return Wwmark(i_file=i, i_mark=m, o_file=o, blind=b, **p).pdf(inspect.stack()[0][3])

    Wwmark(i_file=i, i_mark=m, o_file=o, blind=b, **p).image()


@cli.command(help="Show the blind watermark")
@click.option('-i', help="Your original image path, text blind watermark didn't need it")
@click.option('-m', help="Your blind watermark image", default=None)
@click.option('-o', help="Your output file path")
@click.option('--type', help="Your blind file type, image or text", default="image")
def show(i, m, o, type):
    if str.lower(type) == "image" and i == None:
        # Only image with text blind, it's necessary need original image
        sys.exit(1)

    return Wwmark(i_file=i, i_mark=m, o_file=o, blind=None).show(str.lower(type))


@cli.command(help="clean the watermark, Good effect for translucency watermark")
@click.option('-i', help="Your watermark image path")
@click.option('-m', help="Your blind watermark image", default=None)
@click.option('-o', help="Your output file path")
@click.option('--type', help="Your blind file type, image or text", default=None)
def clean(i, m, o, type):
    if not type:
        type,_,_ = setting(i,None,None)

    return Wwmark(i_file=i, i_mark=m, o_file=o, blind=None).clean(type)

# @click.group(chain=True)
# def cli2():
#     pass

# cli = click.CommandCollection(sources=[cli, cli2])

if __name__ == "__main__":
    cli()
