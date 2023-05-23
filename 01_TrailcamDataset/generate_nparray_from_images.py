import numpy as np
import os
import sys
import glob
import pickle as pkl
import cv2
import argparse
import csv

def get_class(label):
    if label == "No People":
        return 0
    elif label == "Person":
        return 1
    elif label == "Blurry/Bad Picture":
        return 0
    elif label == "Many People":
        return 1
    else:
        return -1

def main(args):
    print(args.data_dir)

    data_dirs = [ "ds1", "ds2", "ds3" ]
    
    image_data = []
    label_data = []
    print(np.shape(image_data))
    print(np.shape(label_data))
    print("^Initial")


    for dir_path in data_dirs:
        
        dir_path += "/IMGs/"

        print(dir_path)

        files = glob.glob( dir_path + "*_result*.csv")

        total_images = 0
        total_persons = 0
        total_no_persons = 0
        total_no_result = 0

        for imagelist_file in files:
            person = 0
            no_person = 0
            no_result = 0
            count_images = 0

            print(imagelist_file)
            count = 0
            fp = open(imagelist_file, "r")
            reader = csv.reader(fp, delimiter=',', dialect='excel')
            next(reader)

            #line = next(reader)

            for line in reader:
                count_images += 1
                label = get_class(line[1])
                if label == 0:
                    no_person += 1
                elif label == 1:
                    person += 1
                elif label == -1:
                    no_result += 1
                
                image = cv2.imread( dir_path + line[0], 0)
                image = np.reshape( image, (1,19200) )
                if label == 0 or label == 1:
                    if len(image_data) == 0:
                        image_data = image
                    else:
                        image_data = np.concatenate( (image_data, image) )
                    label_data = np.append(label_data, label)
                
            print(np.shape(image_data))
            print(np.shape(label_data))
           
            print("Total Images in file: " + str(count_images))
            print("Pictures with people in them: " + str(person) + "(" + str(100*person/count_images) + "%)")
            print("Pictures with no people in them: " + str(no_person) + "(" + str(100*no_person/count_images) + "%)")
            print("Pictures with no classification: " + str(no_result) + "(" + str(100*no_result/count_images) + "%)")

            total_images += count_images
            total_persons += person
            total_no_persons += no_person
            total_no_result += no_result
    
    print("\nTotal Images in all file: " + str(total_images))
    print("Total Pictures with people in them: " + str(total_persons) + "(" + str(100*total_persons/total_images) + "%)")
    print("Total Pictures with no people in them: " + str(total_no_persons) + "(" + str(100*total_no_persons/total_images) + "%)")
    print("Total Pictures with no classification: " + str(total_no_result) + "(" + str(100*total_no_result/total_images) + "%)")

    result = np.true_divide(image_data, 255)

    # Save files to pkl
    np.save(args.output+"-image", image_data)
    np.save(args.output+"-label", label_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'Directory where image data is stored')
    parser.add_argument('--output', type = str, help = 'Name of output file to store pickled data')
    args = parser.parse_args()
    main(args)
    

