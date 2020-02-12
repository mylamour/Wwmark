<div align=center>
<img src="https://user-images.githubusercontent.com/12653147/74099251-93627880-4b5c-11ea-9340-823cbc9eede6.png" />
</div>


# Intro

It based on ffmpeg and opencv. Simply, you can add text or pictures to the target. we support the image, video and pdf to add watermark. Also, original intention is make a mark. So, it should be simply on bottom right, we don't support full screen watermark by default. So you can define it with your local config file with `p` field or use it with '--blind' options. If you want to look more detail, please read the document in FFMPEG homepage. 

| Original / Watermark option | add image | add text | custom position (-p) | invisible(--blind) | clean watermark |
|-----------------------------|-----------|----------|----------------------|--------------------|-----------------|
| image                       | √         | √        | √                    | √                  | √               |
| video                       | √         | √        | √                    | ×                  | x               |
| pdf                         | √         | √        | √                    | x                  | √               |

# Install

* Ubuntu

```bash
sudo apt-get install python3-pip python3-setuptools python3-opencv ffmpeg
pip3 install -r requirements.txt
```

# Useage

<div align=center>

[![asciicast](https://asciinema.org/a/7cwAWEuXm3BN9E10hwhirTm5b.svg)](https://asciinema.org/a/7cwAWEuXm3BN9E10hwhirTm5b)

</div>

```bash

> # echo add image watermark
> python main.py image -i test/h.mp4 -m test/wm.png -o test/h2.mp4
> python main.py image -i test/DLP\ ml.pdf -m test/wm.png -o test/xxx.pdf
> python main.py image -i test/guest.jpg -m test/wm.png -o wi_guest.png
> # echo with blind watermark
> python main.py image -i test/guest.jpg -m test/wm.png -o wi_guest.png --blind
> # also you can add text watermark
> python main.py text -i test/h.mp4 -m "WOWW" -p '{"fontsize":"50"}' -o test/h3.mp4
> python main.py text -i test/guest.jpg -m "人生何处不相逢" -o test/wt_guest.png
> python main.py text -i test/guest.jpg -m "人生何处不相逢" -o test/wt_guest.png --blind
> # if you got blind watermark, may be you want show it
> python main.py show --type text test/wt_guest.png
> python main.py show --type text -i test/wt_guest.png
> python main.py show --type image -i test/wi_guest.png -m test/wm.png -o wm.show.png
> python main.py show --type image -i test/guest.jpg -m test/wi_guest.png -o wm.show.png
> # custom your watermark location
> python main.py image -i test/DLP\ ml.pdf -m test/wm.png -o test/xxx.pdf -p center
> python main.py image -i test/a.pdf -m test/wm.png -o test/test.pdf -p '{"x": "main_w-overlay_w-5","y": "5"}'
> # clean the watermark, it good for alpha < 0.5 watermark. 
> python main.py clean -i test/test.pdf -m test/wm.png -o test/oh.pdf
> python main.py clean -i test/wi_guest.png -o test/oh.png 
```

> it's good way to check your font file in linux system: `fc-list :lang=zh`

## Advance With Config file

Just run `python main.py config -f config.json`

`config.json` 
```json
{
    "action" : "text",
    "i" : "test/guest.jpg",
    "m" : "人生得意须尽欢",
    "o" : "whoareu.jpg",
    "p" : { 
        "x" : "main_w/3",
        "y" : "10",
        "fontsize" : "66",
        "fontfile" : "DroidSansFallbackFull.ttf",
        "box" : "1",
        "boxcolor" : "red"
    }
}
```

# TODO

* [ ] Delete Logo (OCR + Delete Logo)
* [ ] Web UI & Handwriting
* [ ] Human Kindly Output
* [ ] Encrype Sinature Automaticly
* [ ] tests & setup.sh

# Resources

* [关于水印这件“小事”](https://github.com/mylamour/blog/issues/71)
* [how to add transparent watermark](https://stackoverflow.com/questions/10918907/how-to-add-transparent-watermark-in-center-of-a-video-with-ffmpeg)
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)
* [ffmpeg-python docs](https://kkroening.github.io/ffmpeg-python/)
* [ffmpeg simply useage](http://iami.xyz/Image-Parse/)
* [ffmpeg overlay](https://ffmpeg.org/ffmpeg-filters.html#overlay-1)
* [BlindWaterMark](https://github.com/chishaxie/BlindWaterMark)