# Intro

It based on ffmpeg and opencv. There was 3 mode in wwmark. you can add text to a image or video, also you can add image to a image or video. i still try to make it better, like encrypt & decrpt text info to image or make it unvisible. and so on.

# Useage

```bash

i➜  wwmark : master ✘ :✭ ᐅ  python main.py config -f config.json
xxxxxxx

i➜  wwmark : master ✘ :✭ ᐅ  python main.py image -i ../test/video/h.mp4 -m ../test/images/wm.png -o ssss.mp4
xxxxxxx

i➜  wwmark : master ✘ :✭ ᐅ  python main.py image -i ../test/video/h.mp4 -m ../test/images/wm.png -p '{ "x": "main_w-overlay_w-5","y": "5"}'  -o ssss.mp4
xxxxxxx

i➜  wwmark : master ✘ :✭ ᐅ  python main.py text -i ../test/images/guest.jpg -m "人生何处不相逢" -p '{"x":"10", "y":"10","fontsize":"66","fontfile":"DroidSansFallbackFull.ttf"}' -o ian.jpeg
xxxxxxx
```

> it's good way to check your font file in linux system: `fc-list :lang=zh`

# Resources

* [how to add transparent watermark](https://stackoverflow.com/questions/10918907/how-to-add-transparent-watermark-in-center-of-a-video-with-ffmpeg)
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)
* [ffmpeg simply useage](http://iami.xyz/Image-Parse/)
* [ffmpeg overlay](https://ffmpeg.org/ffmpeg-filters.html#overlay-1)