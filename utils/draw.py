import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def draw_ID(img, img_depth, bbox):

    fx = 518.0
    fy = 519.0
    cx = 325.5
    cy = 253.5

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        y = int(( x1 + x2 ) / 2)
        x = int(( y1 + y2 ) / 2)
        depth = img_depth[x, y, 1]
        z_3d = float(depth)
        x_3d = float((x - cx) * z_3d) / fx
        y_3d = float((y - cy) * z_3d) / fy
        position = '[{:.1f}{}{:.1f}{}{:.1f}]'.format(y_3d," ", -x_3d," ", z_3d)

        cv2.putText(img, position, (y, x), cv2.FONT_HERSHEY_PLAIN, 1, [47, 79, 79], 2)
    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
