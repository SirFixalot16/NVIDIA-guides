from ultralytics import YOLO
import cv2

import os
import json
import numpy
import pandas
import csv

def txt_to_csv(txt_filepath
               , csv_filepath
               ):
    """
    Reads a space-delimited txt file and saves it as a CSV file.

    Args:
        txt_filepath (str): Path to the input txt file.
        csv_filepath (str): Path to the output CSV file.
    """
    try:
        df = pandas.read_csv(txt_filepath, sep='\s+', engine='python', header=None)
        print(df)  # Display the first few rows of the DataFrame
        df2 = pandas.DataFrame(numpy.zeros((df.shape[0], 15), dtype=str))
        for index, row in df.iterrows():
            if row[0] == 0.0 or 0: class_name = 'fire'
            elif row[0] == 1.0 or 1: class_name = 'smoke'
            row = [class_name, 0, 0, 0, 
               int((row[1]-row[3]/2)*640),
               int((row[2]-row[4]/2)*640),
               int((row[1]+row[3]/2)*640),
               int((row[2]+row[4]/2)*640),
               0 , 0, 0, 0, 0, 0, 0]
            #print(row)
            df2.loc[index] = row
        print('\n', df2)
        df2.to_csv(csv_filepath, index=False, sep=' ', header=False)
        print(f"Successfully converted '{txt_filepath}' to '{csv_filepath}'")
    except FileNotFoundError:
        print(f"Error: File not found at '{txt_filepath}'")
    except Exception as e:
        print(f"An error occurred: {e}")

def convert_yolo_to_kitti_by_folder(path):
    labels_path = path + '/labels'
    images_path = path + '/images'
    for filename in os.listdir(labels_path):
        if filename.endswith('.txt'):
            txt_filepath = os.path.join(labels_path, filename)
            csv_filepath = os.path.join(labels_path, filename)
            txt_to_csv(txt_filepath, csv_filepath)

def main():
    # txt_to_csv("./fire/test/labels/-FORKLIFT-CATCHES-FIRE-WHILE-OPERATING-IN-A-FACTORY-FIRE-ACCIDENT-CAUGHT-ON-CAMERA_frame_390_jpg.rf.b40cd6ef6607a69a1dc184646dd980a7.txt"
    #            , './test.txt'
    #            )
    convert_yolo_to_kitti_by_folder('./fire/valid')

if __name__ == "__main__":
    main()
