import pandas as pd
import math
import locale
import numpy as np
from scipy import stats
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# mapdata load
def mapdata_load(fileName):
    txtfile = pd.read_csv(fileName, sep=',', encoding="utf-8")
    txtfile = txtfile[['OBJECTID', 'Layer', 'x', 'y', 'Elevation']]
    txtfile['x'] = txtfile['x'].apply(lambda x: locale.atof(x))
    txtfile['y'] = txtfile['y'].apply(lambda x: locale.atof(x))
    line_1 = txtfile.loc[txtfile['Layer'] == "차선_실선"]
    line_2 = txtfile.loc[txtfile['Layer'] == "도로경계"]
    line_3 = txtfile.loc[txtfile['Layer'] == "차선_겹선(3)"]
    line_4 = txtfile.loc[txtfile['Layer'] == "차선_점선"]
    all_line = [line_1, line_2, line_3, line_4]
    return all_line

def calc_dist(point1, point2):
    if len(point1) != len(point2):
        print("point matching error")
        return 0
    else:
        point1 = list(point1)        
        point2 = list(point2)
        if len(point1) == 2:
            dist = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        elif len(point1) == 3:
            dist = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    return dist

def calc_point(pointList, realList, real_newPoint):
    
    tanA = []
    for i in range(len(pointList) - 1):
        j = i + 1
        while j < len(pointList):
            tanA.append(calc_tanA(pointList[i], pointList[j], realList[i], realList[j]))      
            j += 1
    tan = np.mean(tanA)
    Alist = []
    blist = []
    for i in range(len(realList)):
        Alist.append((tan - calc_inclination(realList[i], real_newPoint))/(1+(calc_inclination(realList[i], real_newPoint)*tan)))
        blist.append(pointList[i][1] - (pointList[i][0]*(tan-calc_inclination(realList[i], real_newPoint))/(1+(calc_inclination(realList[i], real_newPoint)*tan))))
    Xlist = []
    Ylist = []
    for i in range(len(pointList) - 1):
        j = i + 1
        while j < len(pointList):
            Xlist.append((blist[i] - blist[j])/(Alist[j] - Alist[i]))
            Ylist.append((Alist[j]*blist[i] - Alist[i]*blist[j])/(Alist[j] - Alist[i]))
            j += 1
    # 중앙값으로 계산
    # px = round(np.median(Xlist))
    # py = round(np.median(Ylist))
    # 절사평균으로 계산
    px = round(stats.trim_mean(Xlist, 0.5))
    py = round(stats.trim_mean(Ylist, 0.5))
    point = (px, py)
    return point

def calc_tanA(point1, point2, realpoint1, realpoint2):
    fr_incl = calc_inclination(point1, point2)
    real_incl = calc_inclination(realpoint1, realpoint2)
    tanA = (fr_incl + real_incl)/(1-(fr_incl*real_incl))
    return tanA

def calc_inclination(point1, point2):
    inclination = (point1[1] - point2[1])/(point1[0] - point2[0])
    return inclination