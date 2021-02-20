import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, apply_classifier
from yolov5.utils.torch_utils import select_device, time_synchronized, load_classifier
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from utilss.draw import draw_boxes, compute_color_for_labels
from utilss.parser import get_config
from utilss.log import get_logger
from utilss.io import write_results
import os
from numpy import random
import numpy as np
import yaml
from collections import Counter
from CoordinateMatching.featureMatching.featMatch import matcher_BRISK_BF
from CoordinateMatching.locMatching.trilateration import point_dist, intersectionPoint, get_trilateration
from mapdata.utils import mapdata_load, calc_dist, calc_point

os.environ['KMP_DUPLICATE_LIB_OK']='True'

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

#Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def compress(data, selectors):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return (d for d, s in zip(data, selectors) if s)

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, GCP_list = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.GCP_list 
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path_raw = str(Path(out)) + '/results_raw.txt'
    
    # 속도까지 붙여버린 데이터 따로 생성해서 비교해보자 : 수정수정
    txt_path_raw2 = str(Path(out)) + '/results_raw2.txt'

    # point load
    with open('./mapdata/point.yaml') as f:
        data = yaml.load(f.read())    
    frm_point = data['frm_point']
    geo_point = data['geo_point']

    Counter_1 = [(488,589), (486,859)]
    Counter_2 = [(3463,795), (3487,1093)]
    Counter_list = [Counter_1, Counter_2]

    datum_dist = []
    counter_dist = []
    
    line_fileName = './mapdata/Busan1_IC_Polyline_to_Vertex.txt'
    all_line = mapdata_load(line_fileName, frm_point, geo_point)

    percep_frame = 5
    from _collections import deque
    pts = [deque(maxlen=percep_frame+1) for _ in range(10000)]
    ptsSpeed = [deque(maxlen=1) for _ in range(10000)]

    frame_len = calc_dist(frm_point[1], frm_point[4])
    geo_len = calc_dist(geo_point[1], geo_point[4])

    # ----------------- fix val start
    fixcnt = 1
    # ----------------- fix val end

    # ----------------- counter val start
    memory_index = {}
    memory_id = {}

    cnt = np.zeros((len(Counter_list),4))
    # total_counter = 0 # 나중에 총 카운터를 만들어 넣으면 되겠지?

    # count_1_total = 0
    # count_1_veh_c0 = 0
    # count_1_veh_c1 = 0
    # count_1_veh_c2 = 0

    # count_2_total = 0
    # count_2_veh_c0 = 0
    # count_2_veh_c1 = 0
    # count_2_veh_c2 = 0
    # ----------------- counter val end

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            print(pred)

        t2 = time_synchronized()   

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                clss = []
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    clss.append(cls.item())

                bbox_xywh = bbox_xywh
                cls_conf = confs
                cls_ids = clss
                # xywhs = torch.Tensor(bbox_xywh)
                # confss = torch.Tensor(confs)
                # cls_ids = clss

                # if len(bbox_xywh) == 0:
                #     continue
                # print("detection cls_ids:", cls_ids)

                #filter cls id for tracking
                # print("cls_ids")
                # print(cls_ids)

                # # select class
                # mask = []
                # lst_move_life = [0,1,2]
                # # lst_for_track = []
                
                # for id in cls_ids:
                #     if id in lst_move_life:
                #         # lst_for_track.append(id)
                #         mask.append(True)
                #     else:
                #         mask.append()
                # # print("mask cls_ids:", mask)

                # # print(bbox_xywh)
                # bbox_xywh = list(compress(bbox_xywh,mask))
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                # bbox_xywh[:,3:] *= 1.2
                # cls_conf = list(compress(cls_conf,mask))
                # print(cls_conf)

                bbox_xywh = torch.Tensor(bbox_xywh)
                cls_conf = torch.Tensor(cls_conf)

                # Pass detections to deepsort
                outputs = deepsort.update(bbox_xywh, cls_conf, im0, cls_ids)

                """
                # output 형식

                [[박스 좌측상단 x, 박스 좌측상단 y, 박스 우측하단 x, 박스 우측하단 y, 차량 id, 클래스 넘버],
                [박스 좌측상단 x, 박스 좌측상단 y, 박스 우측하단 x, 박스 우측하단 y, 차량 id, 클래스 넘버],
                [박스 좌측상단 x, 박스 좌측상단 y, 박스 우측하단 x, 박스 우측하단 y, 차량 id, 클래스 넘버],
                [박스 좌측상단 x, 박스 좌측상단 y, 박스 우측하단 x, 박스 우측하단 y, 차량 id, 클래스 넘버],
                ...]
                """

# ------------------------------------------------------------------------------------------------------ img fix start
                t3 = time_synchronized()
                match_mid_point_list = matcher_BRISK_BF(im0, GCP_list)
                t4 = time_synchronized()
# ---------------------------------------------------------------------------------------------------------------------- line start

                # 기준점 위치 갱신을 위한 삼변측량의 거리 정의 및 고정
                if frame_idx == 0:
                    for pointNum in range(len(frm_point)):
                        for GCP_num in range(len(match_mid_point_list)):
                            datum_dist.append(point_dist(match_mid_point_list[GCP_num], frm_point[pointNum]))
                    datum_dist = np.reshape(datum_dist,(len(frm_point),len(match_mid_point_list)))
                    for Ct_list in Counter_list:
                        for Ctpoint_num in range(len(Ct_list)):
                            for GCP_num in range(len(match_mid_point_list)):
                                counter_dist.append(point_dist(match_mid_point_list[GCP_num], Ct_list[Ctpoint_num]))
                    counter_dist = np.reshape(counter_dist,(len(Counter_list),len(Ct_list), len(match_mid_point_list)))
                t5 = time_synchronized()

                pre_P = (0,0)

                for line_num, eachline in enumerate(all_line):
                    for newpoint in eachline['frmPoint']:
                        if line_num == 0:
                            im0 = cv2.circle(im0, newpoint, 5, (0,0,255),-1)    # 차선_실선
                            if calc_dist(pre_P, newpoint) < 390:
                                im0 = cv2.line(im0, pre_P, newpoint, (0,0,255), 2,-1)
                        elif line_num == 1:
                            im0 = cv2.circle(im0, newpoint, 5, (0,255,0),-1)    # 도로 경계
                            if calc_dist(pre_P, newpoint) < 420:
                                im0 = cv2.line(im0, pre_P, newpoint, (0,255,0), 2,-1)
                        elif line_num == 2:
                            im0 = cv2.circle(im0, newpoint, 5, (255,0,0),-1)    # 차선_겹선
                            if calc_dist(pre_P, newpoint) < 350:
                                im0 = cv2.line(im0, pre_P, newpoint, (255,0,0), 2,-1)
                        else:
                            im0 = cv2.circle(im0, newpoint, 5, (100,100,0),-1)  # 차선_점선
                            if calc_dist(pre_P, newpoint) < 600:
                                im0 = cv2.line(im0, pre_P, newpoint, (100,100,0), 2,-1)
                        pre_P = newpoint

                t6 = time_synchronized()
                for pointNum in range(len(frm_point)):
                    im0 = cv2.circle(im0, frm_point[pointNum], 10, (0,0,0),-1)
                    newPoint = intersectionPoint(match_mid_point_list, datum_dist[pointNum])
                    frm_point[pointNum] = newPoint

                t7 = time_synchronized()

#---------------------------------------------------------------------------------------------------------------------- line end                

# ------------------------------------------------------------------------------------------------------ img fix end

# ------------------------------------------------------------------------------------------------------ counting num and class start
                Counter_newpoint = []
                for Ct_num in range(len(Counter_list)):
                    Ct_list = Counter_list[Ct_num]
                    for Ctpoint_num in range(len(Ct_list)):
                        Counter_newpoint.append(intersectionPoint(match_mid_point_list, counter_dist[Ct_num][Ctpoint_num]))
                Counter_newpoint = np.reshape(Counter_newpoint, (len(Counter_list), len(Ct_list), 2))
                
                for CountNum in Counter_newpoint:
                    im0 = cv2.line(im0, tuple(CountNum[0]), tuple(CountNum[1]), (0,0,0), 5,-1)                

                boxes = []
                indexIDs = []
                classIDs = []
                previous_index = memory_index.copy()
                previous_id = memory_id.copy()
                memory_index = {}
                memory_id = {}
                COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        boxes.append([output[0], output[1], output[2], output[3]])
                        indexIDs.append(int(output[4]))
                        classIDs.append(int(output[5]))
                        memory_index[indexIDs[-1]] = boxes[-1] # 인덱스 아이디와 박스를 맞춰줌
                        memory_id[indexIDs[-1]] = classIDs[-1] # 인덱스 아이디와 클레스 아이디를 맞춰줌

                        if len(pts[output[4]]) == 0:
                            pts[output[4]].append(frame_idx)
                        center = (int(((output[0]) + (output[2]))/2), int(((output[1]) + (output[3]))/2))
                        pts[output[4]].append(center)
                        if len(pts[output[4]]) == percep_frame + 1:
                            frmMove_len = np.sqrt(pow(pts[output[4]][-1][0] - pts[output[4]][-percep_frame][0], 2) + pow(pts[output[4]][-1][1] - pts[output[4]][-percep_frame][1], 2))
                            geoMove_Len = geo_len * frmMove_len / frame_len
                            speed = geoMove_Len * vid_cap.get(cv2.CAP_PROP_FPS) * 3.6 / (pts[output[4]][0]-frame_idx)
                            ptsSpeed[output[4]].append(speed)
                            pts[output[4]].clear()

                if len(boxes) > 0:
                    i = int(0)
                    for box in boxes:
                        # 현 위치와 이전 위치를 비교하여 지나갔는지 체크함
                        (x, y) = (int(box[0]), int(box[1])) # Output 0 1 
                        (w, h) = (int(box[2]), int(box[3])) # Output 2 3 과 같다.
                        color = compute_color_for_labels(indexIDs[i])

                        if indexIDs[i] in previous_index:
                            previous_box = previous_index[indexIDs[i]]
                            # print()
                            # print('previous_box : ')
                            # print(previous_box)
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2)) # 현재 박스
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2)) # 이전 박스
                            cv2.line(im0, p0, p1, color, 3) # 이전 정보와 비교 : 중앙에 점을 찍어 가면서 (이전 데이터와 검지 데이터의 점)

                            # 클레스 구분
                            previous_class_id = previous_id[indexIDs[i]] # 어차피 인덱스 같기 때문에 그냥 넣어줘도 됨 개꿀ㅋ


                            # Yolov5와 DeepSort를 통하여 만들어진 첫 결과물(내가 맨든 결과물)
                            # 프레임 수, 인덱스 아이디, 클레스 이름, x좌표, y좌표, w값, h값, 속도값, null, null
                            # with open(txt_path_raw2, 'a') as f:
                            #     f.write(('%g ' * 10+ '\n') % (frame_idx, indexIDs[i], previous_class_id,
                            #                                 p0[0], p0[1], box[2], box[3], -1, -1))  # label format

                            for cntr in range(len(Counter_newpoint)):
                                if intersect(p0, p1, Counter_newpoint[cntr][0], Counter_newpoint[cntr][1]): # 실질적으로 체크함
                                    if previous_class_id == 0 : cnt[cntr][1] += 1
                                    elif previous_class_id == 1 : cnt[cntr][2] += 1
                                    elif previous_class_id == 2 : cnt[cntr][3] += 1
                                    cnt[cntr][0] += 1

                        i += 1 # 다음 인덱스와 비교하게 만들기 위하여
        
                
                # draw counter
                for cntr in range(len(Counter_newpoint)):
                    cv2.putText(im0, 'count_{}_total : {}'.format(cntr+1, cnt[cntr][0]), (100+400*cntr, 110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2) # 카운팅 되는거 보이게
                    cv2.putText(im0, 'count_{}_{} : {}'.format(cntr+1, names[0], cnt[cntr][1]), (100+400*cntr, 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2) # 카운팅 되는거 보이게
                    cv2.putText(im0, 'count_{}_{} : {}'.format(cntr+1, names[1], cnt[cntr][2]), (100+400*cntr, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2) # 카운팅 되는거 보이게
                    cv2.putText(im0, 'count_{}_{} : {}'.format(cntr+1, names[2], cnt[cntr][3]), (100+400*cntr, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2) # 카운팅 되는거 보이게
                t8 = time_synchronized()
# ---------------------------------------------------------------------------------------------------------------------- counter end

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4:5]
                    cls_id = outputs[:,-1]
                    draw_boxes(im0, bbox_xyxy, cls_id, identities, names, ptsSpeed)

                t9 = time_synchronized()
                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs): # 한 라인씩 쓰는 구조
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[4]
                        classname = output[5]

                        with open(txt_path_raw, 'a') as f: # Yolov5와 DeepSort를 통하여 만들어진 첫 결과물(원본결과물)
                            f.write(('%g ' * 6 +'%g' *1 +'%g ' * 3 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, classname, -1, -1, -1))  # label format

            # else:
            #     deepsort.increment_ages()
            t10 = time_synchronized()
            # Print time (inference + NMS + classify)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            t11 = time_synchronized()
            # Save results (image with detections)
            # dataset.mode = 'images'
            # save_path = './track_result/output/{}.jpg'.format(i)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
            t12 = time_synchronized()

            print('inference + NMS + classify (%.3fs)' % (t2 - t1))
            print('Yolo + DeepSORT (%.3fs)' % (t3 - t2))
            print('find mid point (%.3fs)' % (t4 - t3))
            print('삼변측량을 위한 기준거리 산정 (%.3fs)' % (t5 - t4))
            print('draw line (%.3fs)' % (t6 - t5)) # 현재는 정밀도로지도에 있는 모든 점들을 대상 계산중 -> 추후 화면에 표시될 점만 계산하는 작업 필요
            print('GCP 점 계산 (%.3fs)' % (t7 - t6))
            print('Count & speed (%.3fs)' % (t8 - t7))
            print('각차량별 그리기 (%.3fs)' % (t9 - t8))
            print('txt 데이터 저장 (%.3fs)' % (t10 - t9))
            print('스크린에 표시하기 (%.3fs)' % (t11 - t10))
            print('비디오파일로 저장하기 (%.3fs)' % (t12 - t11))
            print('one frame done (%.3fs)' % (t12 - t1))

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':

    # 이제 트레킹을 시작해볼까
    img_size = 800 # 이미지 사이즈(default : 640) : 이미지의 크기를 조절(resizing)하여 검출하도록 만듦, 크면 클수록 검지율이 좋아지지만 FPS가 낮아짐, 내 모델에서는 800*800 기준으로 훈련했단다..아직은.. 곧 새컴옴 ㅅㄱ
    model_weights = 'train_result/20210208/weights/best.pt' # custom 모델 경로를 맞추는 것은 기본이겠지?
    test_data_path = 'input_video' # 테스트 비디오 경로를 맞추는 것도 기본이겠지?
    classes_type = [0, 1, 2] # 내가 할땐 차량을 3가지로 구분하였단다.. Lette is horse...
    conf_thres = 0.4
    iou_thres = 0.5
    parser = argparse.ArgumentParser()
    # ./CoordinateMatching/pointcapturing.py 에서 만들어놨던 이미지를 경로로 설정해주면 되겠지?
    GCP_list = ['./CoordinateMatching/matchimg/cap_source/cap_result/point_2_0167_3413_1751.jpg',
                './CoordinateMatching/matchimg/cap_source/cap_result/point_3_0167_1393_287.jpg',
                './CoordinateMatching/matchimg/cap_source/cap_result/point_4.jpg']
                # 1번 5번 점이 불안함 -> 담에 돌릴때는 두개 빼고 하자

    parser.add_argument('--GCP_list', type=str,
                        default=GCP_list, help='GCP_point_path_list')
    parser.add_argument('--weights', type=str,
                        default=model_weights, help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default=test_data_path, help='source')
    parser.add_argument('--output', type=str, default='track_result/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=img_size,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=conf_thres, help='object confidence threshold') # 중복 제거의 수준
    parser.add_argument('--iou-thres', type=float,
                        default=iou_thres, help='IOU threshold for NMS') # 검출 박스 IOU(교집할) 설정
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)') # 저장 형식
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true', default='track_result/output',
                        help='save results to *.txt') # 결과 택스트 파일로 저장 output.txt.로 저장됨
    parser.add_argument('--classes', nargs='+', type=int,
                        default=classes_type, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
