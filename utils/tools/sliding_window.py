import numpy as np
import math
import cv2
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import time

def get_window_length(l,w,padding):
    # Get max value
    val_list = [float(l),float(w)]
    max_side = max(val_list)
    # print(f'Max value is {max_side}')

    # Calculate value of interior square side (i_len)
    i_len = max_side + 2*padding

    # Calculate value of exterior square side (e_len)
    e_len = (2*(math.sqrt(((i_len)**2))/2))
    print(f'\nThe value of the square window side is {e_len} meters.\n')
    return e_len


def sliding_window(window_len, pcd_shape, step):
    pcd_l = pcd_shape[1]
    pcd_w = pcd_shape[0]
    print(f'\nAnalyzed area is {pcd_w} meters wide and {pcd_l} meters long.\n')
    x_flag = True
    x_counter = 0
    y_flag = True
    y_counter = 0
    window_list = []


    stop_flag = False
    while(True):
        w_y_min = step*y_counter
        w_y_max = w_y_min + window_len
        if w_y_max == pcd_l:
            stop_flag = True
            y_counter = 0
        elif w_y_max>pcd_l:
            stop_flag = True
            w_y_max = pcd_l
            w_y_min = w_y_max - window_len
            y_counter = 0
        else:
            y_counter += 1
        while(True):
            w_x_min = step*x_counter
            w_x_max = w_x_min + window_len
            if w_x_max == pcd_w:
                window_data = [[w_x_min, w_x_max],[w_y_min, w_y_max]]
                window_list.append(window_data)  
                x_counter = 0
                break
            elif w_x_max > pcd_w:
                w_x_max = pcd_w
                w_x_min = w_x_max - window_len
                window_data = [[w_x_min, w_x_max],[w_y_min, w_y_max]]
                window_list.append(window_data)  
                x_counter = 0
                break
            else:
                x_counter += 1
                window_data = [[w_x_min, w_x_max],[w_y_min, w_y_max]]
                window_list.append(window_data)
        if stop_flag:
            break


    img = np.zeros((2000,2000,3), np.uint8)
    img2 = cv2.imread('samples/png/munition_sample.png')
    # blank_image = np.zeros((9.82455778,6.55983877,3), np.uint8)
    crop_img = img2[0:799, 0:499]
    img2 = crop_img
    time.sleep(6)


    prev_x_min = 0
    prev_x_max = 1
    prev_y_min = 0
    prev_y_max = 1
    for i,window in enumerate(window_list):
        x_min = int(window[0][0]*10)
        x_max = int(window[0][1]*10)
        y_min = int(window[1][0]*10)
        y_max = int(window[1][1]*10)
        print('\n')
        print('--------------------------------')
        print(f'Sample no. {i}')
        print('Min x: ', x_min/10, 'Max x: ', x_max/10)
        print('Min y: ', y_min/10,'Max y: ', y_max/10)
        print('--------------------------------')
        print('\n')
        if i != 0:
            cv2.rectangle(img2,(prev_x_min, prev_y_min),(prev_x_max,prev_y_max),(0,0,255),3)
        cv2.rectangle(img2,(x_min, y_min),(x_max,y_max),(0,255,0),3)
        cv2.imshow("test image", img2)
        filename = 'animations/1/frame_'+str(i) + '.jpg'
        cv2.imwrite(filename, img2)
        cv2.waitKey(100)
        prev_x_min = x_min
        prev_x_max = x_max
        prev_y_min = y_min
        prev_y_max = y_max

    cv2.imshow("test image", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # l = input('Input length: ')
    # w = input('Input width: ')
    l = 5
    w = 10
    print(f'\nSearching for objects {l} meters long and {w} meters wide.\n')
    padding = 1
    # padding = .2
    window_len = get_window_length(l, w, padding)
    print('')
    # slide_window(window_len, [10,15,5], step=1)
    sliding_window(window_len, [50,80,5], step=5)