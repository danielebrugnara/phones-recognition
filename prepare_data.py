# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import python_speech_features as psf
from scipy.io.wavfile import read
import argparse
import numpy as np
import pickle as pkl
import os

sampling = 16000.    
suffix="phn"

def compute_mfcc(file, FLAGS):
    print("converting file %s" % (file))
    data = np.array(read(file)[1], dtype=float)
    return psf.base.mfcc(signal=data, winstep=FLAGS.frame_step, winlen=FLAGS.frame_length,
                         winfunc=np.hamming)

l2i={}

def readl2i(fname='/home/fabio/sw/TensorFlow/projects/timit-fromscratch/ph2idx'):
    global l2i
    l2i={}
    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            (l, i) = line.split()
            l2i[l]=i
            
def get_label(file, length, FLAGS):
    # print ('get_label', file)
    with open(os.path.join(os.path.splitext(file)[0]+"."+suffix)) as f:
        lines = f.readlines()
        labels = np.zeros(length)
        lines = [x.strip() for x in lines]
        i=0
        time=0
        step=sampling*FLAGS.frame_step
        for line in lines:
            #
            # print (line)
            (start,end,lbl) = line.split()
            end=float(end)
            end=np.floor(end/step)*step
            while time<end and i<length:
                # print (time, i, step, l2i[lbl])
                labels[i] = l2i[lbl]
                time+=step
                i+=1
        if i > len(labels):
            print ('length mismatch: ', i, len(labels))
            sys.exit(1)
    return labels


def main(FLAGS):
    
    file_number = 0 
    readl2i()
    # print (l2i)
    with open(FLAGS.file_list_file) as f:
        files = f.readlines()
        files = [x.strip() for x in files]
        for file in files:
            file_number +=1
            data_tmp = compute_mfcc(file, FLAGS)  
            label_tmp = get_label(file, data_tmp.shape[0], FLAGS)
            if file_number==1:
                data = data_tmp
                label = label_tmp
            else:
                data = np.concatenate((data,data_tmp))
                label = np.concatenate((label,label_tmp))

            print("Data size %s shape %s " % (data.size, data.shape))
            #print(data.shape)
    np.save(FLAGS.data_output_name, data)
    np.save(FLAGS.label_output_name, label)

     
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list_file', type=str,
                        default='/home/fabio/sw/TensorFlow/projects/timit-fromscratch/trainlist',
                        help="file containing list of files to convert")
    parser.add_argument('--output_file', type=str, 
                        help="name of the output pickle file")
    parser.add_argument('--frame_step', type=float,
                        help="step of the MFCC framing", default=0.01)
    parser.add_argument('--frame_length', type=float,
                        help="length of the frame", default=0.020)
    parser.add_argument('--data_output_name', type=str,
                        help="name of the output data file", default='/home/fabio/sw/TensorFlow/projects/timit-fromscratch/out_data' )
    parser.add_argument('--label_output_name', type=str,
                        help="name of the output label file", default='/home/fabio/sw/TensorFlow/projects/timit-fromscratch/out_label')
    FLAGS = parser.parse_args()
    main(FLAGS)
