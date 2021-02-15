import math
from scipy.optimize import least_squares

# 점1 x,y 와 점2 x,y의 거리 구하는 공식
def point_dist(x1,y1,x2,y2):
    
    a = x2 - x1   # 선 a의 길이
    b = y2 - y1    # 선 b의 길이
    
    dist = math.sqrt((a * a) + (b * b))    # (a * a) + (b * b)의 제곱근을 구함
    return dist


def intersectionPoint(point_list, dist_list):

    pointX = []
    pointY = []
    
    for pointNum in range(len(point_list)):
        pointX.append(point_list[pointNum][0])
        pointY.append(point_list[pointNum][1])

    def eq(g):
        x, y = g
        
        pointResult = []
        for pointNum in range(len(pointX)):
            pointResult.append((x - pointX[pointNum])**2 + (y - pointY[pointNum])**2 - dist_list[pointNum]**2)
        pointResult = tuple(pointResult)
        return pointResult

    guess = (pointX[0], pointY[0] + dist_list[0])

    ans = least_squares(eq, guess, ftol=None, xtol=None)
    target_point = (round(ans.x[0]), round(ans.x[1]))

    return target_point