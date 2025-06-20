#!/usr/bin/env python

import os
import ivport
import time

    
def picam_sequence_4():
    
    def sequence_outputs(iv):
        for i in range(3):
                iv.camera_change(1)
                iv.camera_capture("NAME" + str(i), use_video_port=False)
                time.sleep(2)
                iv.camera_change(2)
                iv.camera_capture("NAME" + str(i), use_video_port=False)
                time.sleep(2)
                iv.camera_change(3)
                iv.camera_capture("NAME" + str(i), use_video_port=False)
                time.sleep(2)
                iv.camera_change(4)
                iv.camera_capture("NAME" + str(i), use_video_port=False)
                time.sleep(3)   # SD Card Bandwidth Correction Delay
    iv = ivport.IVPort(ivport.TYPE_QUAD2, iv_jumper='A')
    iv.camera_open(camera_v2=True, resolution=(2592, 1944), framerate=10)
    iv.camera_sequence(outputs=sequence_outputs(iv), use_video_port=True)
    iv.close()    
    

# main capture examples
# all of them are functional
def main():
    #still_capture()
    #picam_capture()
    #picam_sequence()
    picam_sequence_4()

if __name__ == "__main__":
    main()
