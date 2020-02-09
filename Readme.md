# Intro

It based on ffmpeg and opencv. There was 3 mode in wwmark. you can add text to a image or video, also you can add image to a image or video. i still try to make it better, like encrypt & decrpt text info to image or make it unvisible. and so on.

# Useage

```bash

i➜  wwmark : master ✘ :✭ ᐅ  python main.py config -f config.json

i➜  wwmark : master ✘ :✭ ᐅ  python main.py image -i ../test/video/h.mp4 -m ../test/images/wm.png -o ssss.mp4

i➜  wwmark : master ✘ :✭ ᐅ  python main.py image -i ../test/video/h.mp4 -m ../test/images/wm.png -p '{ "x": "main_w-overlay_w-5","y": "5"}'  -o ssss.mp4

i➜  wwmark : master ✘ :✭ ᐅ  python main.py text -i ../test/images/guest.jpg -m "人生何处不相逢" -p '{"x":"10", "y":"10","fontsize":"66","fontfile":"DroidSansFallbackFull.ttf"}' -o ian.jpeg

i➜  wwmark : master ✘ :✭ ᐅ  python main.py text -i test/a.pdf -m "人生何处不相逢" -o b.pdf

i➜  wwmark : master ✘ :✭ ᐅ  python main.py text -i test/a.pdf -m "人生何处不相逢" -o b.pdf

i➜  wwmark : master ✘ :✹ ᐅ  python main.py text -i test/images/guest.jpg -m "人生何处不相逢" -o here.png --blind

i➜  wwmark : master ✘ :✹ ᐅ  python main.py image -i test/images/guest.jpg -m test/images/wm.png -o wocao.png --blind

i➜  wwmark : master ✘ :✹ ᐅ  python main.py show --type text -i here.png

i➜  wwmark : master ✘ :✹ ᐅ  python main.py show --type image -i test/images/guest.jpg -m wocao.png -o test.jpg

```

> it's good way to check your font file in linux system: `fc-list :lang=zh`

# Resources

* [how to add transparent watermark](https://stackoverflow.com/questions/10918907/how-to-add-transparent-watermark-in-center-of-a-video-with-ffmpeg)
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)
* [ffmpeg simply useage](http://iami.xyz/Image-Parse/)
* [ffmpeg overlay](https://ffmpeg.org/ffmpeg-filters.html#overlay-1)
* [BlindWaterMark](https://github.com/chishaxie/BlindWaterMark)