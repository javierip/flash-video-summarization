# Flash Video Summarization
A faster than realtime video summarization algorithm
# Compile from source:
```
$ git clone https://github.com/javierip/flash-video-summarization.git
$ cd flash-video-summarization
$ cd source
$ run.sh
```

# Tested enviroments
* Linux Mint 18 (x64)
* OpenCV 3.0.0


## Available options:
Usage: flash_summ [params] 

        -d, --distance (value:0.2)
                matches distance threshold
        -e, --step (value:3)
                distance between two processed frames
        -g, --gui
                display video and GUI interface
        -h, --help (value:true)
                print help message
        -i, --input (value:video.mpg)
                specify input video
        --interval, -t (value:30)
                interval for average matches count
        -n, --noise (value:0.8)
                noise threshold
        -o, --output (value:./output/)
                output folder path
        -s, --sensitivity (value:0.4)
                sensitivity threshold
        -v, --vervose
                print internal values

Example of usage:


./flash_summ -i=../../data/v21.mpg -o=../../output/v21 -s=0.4 -n=0.96 -d=0.25 -t=30 -e=3



# Citation
If case that you use this source code. Please [cite this paper](http://ieeexplore.ieee.org/document/6746822/)

```
@INPROCEEDINGS{6746822, 
author={J. Iparraguirre and C. Delrieux}, 
booktitle={2013 IEEE International Symposium on Multimedia}, 
title={Speeded-Up Video Summarization Based on Local Features}, 
year={2013}, 
volume={}, 
number={}, 
pages={370-373}, 
keywords={feature extraction;object tracking;video signal processing;digital video;feature tracking;local features;speeded-up video summarization;uncompressed domain;video data;video stream;Color;Feature extraction;Histograms;Noise;Proposals;Standards;Streaming media;Video summarization;keyframe selection;local features;video processing;video skimming}, 
doi={10.1109/ISM.2013.70}, 
ISSN={}, 
month={Dec},}
```
