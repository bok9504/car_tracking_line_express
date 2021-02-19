import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, cls_id, identities=None, names=[], ptsSpeed=[], offset=(0,0)):
    for i,box in enumerate(bbox):

        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        id_num = '{}{:d}'.format("", id)
        clss = names[cls_id[i]]
        color = compute_color_for_labels((cls_id[i] + 1)*2)
        if len(ptsSpeed[id]) == 0:
            label = "{}-{}".format(clss, id_num) # 이걸로 cls_id 표출함
        else:
            label = "{}-{} Speed:{}".format(clss, id_num, int(abs(ptsSpeed[id][0]))) # 이걸로 cls_id 표출함
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]

        cv2.rectangle(img,(x1, y1),(x2,y2),color,3) # x1,y1 : 스타트 위치, x2, y2 : 엔드 위치 --> 중앙 좌표값이 아닌 좌상우하다
        cv2.rectangle(img,(x1+(t_size[0]+3), y1-(t_size[1]+4)),(x1,y1), color,-1)
        cv2.putText(img,label,(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], 2)
    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
