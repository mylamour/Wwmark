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
                Wwmark(i_file=c['i'], i_mark=c['m'], o_file=c['o'], **c['p']).text()
            elif str.lower(c['action']) == "image":
                Wwmark(i_file=c['i'], i_mark=c['m'], o_file=c['o'], **c['p']).image()

@cli.command()
@click.option('-i', help="Your input file path, it can be image or video")
@click.option('-m', help="Your mark file path, it only can be text")
@click.option('-o', help="Your output file path, ")
@click.option('-p', help='mark postion, with like this { "x": "15","y": "5"}', default=OverlayText.DEFAULT.value)
@click.option('-b', help='Blind or not?', default=False)
def text(i, m, o, p, b):
    if p != OverlayText.DEFAULT.value:
        p = json.loads(p)

    if mime.from_file(i) == "application/pdf":
        return Wwmark(i_file=i, i_mark=m, o_file=o, **p).pdf(inspect.stack()[0][3])

    Wwmark(i_file=i, i_mark=m, o_file=o, **p).text()

@cli.command()
@click.option('-i', help="Your input file path, it can be image or video")
@click.option('-m', help="Your mark file path, it can be image or video")
@click.option('-o', help="Your output file path, ")
@click.option('-p', help='mark postion, with like this {"x": "main_w-overlay_w-5","y": "5"}', default=OverlayImage.DEFAULT.value)
@click.option('-b', help='Blind or not?', default=False)
def image(i, m, o, p, b):

    if p != OverlayImage.DEFAULT.value:
        p = json.loads(p)

    if mime.from_file(i) == "application/pdf":
        return Wwmark(i_file=i, i_mark=m, o_file=o, **p).pdf(inspect.stack()[0][3])

    Wwmark(i_file=i, i_mark=m, o_file=o, **p).image()


if __name__ == "__main__":
    cli()
