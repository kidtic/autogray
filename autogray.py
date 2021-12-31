# -*- coding: utf-8 -*- 
import cv2
import numpy as np
import os
import math
import sys
import matplotlib.pyplot as plt

g_drcValue = 50
#全图宽长度 um
g_imgLen = 1400

#画比例尺
def draw_scale(srcimg):
    pass

def my_cv_imread(filepath):
    # 使用imdecode函数进行读取
    retimg = cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    return retimg

#调整图像为100的亮度
def modifyImgBeta(srcimg,bx):
    ele=cv2.mean(srcimg)
    srcbrigh = (ele[0]+ele[1]+ele[2])/3
    diff = bx - srcbrigh
    return cv2.addWeighted(srcimg,1,np.zeros(srcimg.shape,srcimg.dtype),0,diff)

# 读取目录下所有的jpg图片
def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

def sobelimg(srcimg):
    x = cv2.Sobel(srcimg, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(srcimg, cv2.CV_16S, 0, 1)
    # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)

def showplt(nparr):
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams.update({"font.size": 8})
    x = range(len(nparr))
    plt.plot(x, nparr)
    plt.show()

#得到低于value的中间带(滞回比较)
def getBigDiff(nparr,value):
    sidepix = 50
    minr=0;minl=0
    for i in range(sidepix,len(nparr)-sidepix,1):
        if nparr[i]<value:
            minl=i;break
    for i in range(len(nparr)-sidepix,sidepix,-1):
        if nparr[i]<value:
            minr=i;break
    return minr,minl

def decteLine_cmp(srcimg):
    global g_drcValue
    dstgray = cv2.equalizeHist(srcimg)
    dstdf = sobelimg(dstgray)
    #计算每列的平均值
    meanA_row = dstdf.mean(axis=0) 
    diff_meanA_row = np.convolve(meanA_row, np.ones((10,))/10,"same")
    minr,minl = getBigDiff(diff_meanA_row,g_drcValue)
    return minr,minl

def decteLine(srcimg):
    dstdf = sobelimg(srcimg)
    #计算每列的平均值
    meanA_row = dstdf.mean(axis=0) 
    sumAll = meanA_row.sum()
    #计算左区域的求和分布
    sumLeft_all = np.zeros(len(meanA_row))
    e_sum=0
    for i in range(len(meanA_row)):
        e_sum = e_sum + meanA_row[i]
        sumLeft_all[i] = e_sum
    #计算右边区域的求和分布
    sumRight_all = np.zeros(len(meanA_row))
    e_sum=0
    for i in range(len(meanA_row)-1,-1,-1):
        e_sum = e_sum + meanA_row[i]
        sumRight_all[i] = e_sum
    #计算左画线与
    rowsize = len(meanA_row)
    mindiff = -1000000;minr=0;minl=0
    for l in range(rowsize//2):
        for r in range(l+20,rowsize-20,1):
            sumM = sumAll-sumRight_all[r]-sumLeft_all[l]
            sumSide = sumAll-sumM
            meanM = sumM/(r-l-1)
            meanSide = sumSide/(rowsize-r+l+1)
            delta = meanSide-meanM
            if delta>mindiff:
                mindiff=delta;minr=r;minl=l
    return minr,minl


if __name__ == "__main__" and len(sys.argv)==1:
    #判断dst与src目录是否存在
    if os.path.exists("src") == False:
        print("error: src目录不存在")
    if os.path.exists("dst") == False:
        os.mkdir("dst") 
    if os.path.exists("dstgray") == False:
        os.mkdir("dstgray") 
    if os.path.exists("dstline") == False:
        os.mkdir("dstline") 

    file_name=list_all_files("src")
    for imgdir in file_name:
        
        img = my_cv_imread(imgdir)
        ele=cv2.mean(img);brigh = (ele[0]+ele[1]+ele[2])/3
        dst = modifyImgBeta(img,105)
        ele=cv2.mean(dst);brigh = (ele[0]+ele[1]+ele[2])/3
        basefile_name = os.path.basename(imgdir)
        cv2.imencode('.jpg', dst)[1].tofile("dst/"+basefile_name)

        dstgray_c = cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(3,(8,8))
        dstgray = clahe.apply(dstgray_c)
        #dstgray = cv2.equalizeHist(dstgray_c)
        
        #print((dstgray.dtype))
        cv2.imencode('.jpg', dstgray)[1].tofile("dstgray/"+basefile_name)

        #高斯滤波 图像处理
        #dstgray = cv2.equalizeHist(dstgray_c)
        #dstgray=cv2.GaussianBlur(dstgray,(7,7),0)
        #检测划线
        liner,linel = decteLine_cmp(dstgray_c)
        ptStart = (linel, 0)
        ptEnd = (linel, dstgray.shape[0])
        cv2.line(dstgray, ptStart, ptEnd, 255, 8, 4)
        ptStart = (liner, 0)
        ptEnd = (liner, dstgray.shape[0])
        cv2.line(dstgray, ptStart, ptEnd, 255, 8, 4)
        cv2.imencode('.jpg', dstgray)[1].tofile("dstline/"+basefile_name)
        print(imgdir+"  w: "+str(liner-linel)+"pix")
    cv2.waitKey(0)
    
        
if __name__ == "__main__" and len(sys.argv)==2 and sys.argv[1]=="test":
    print("exetest")
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #判断dst与src目录是否存在
    if os.path.exists("src") == False:
        print("error: src目录不存在")
    file_name=list_all_files("src")
    if(len(file_name)==0):
        print("no img")
        sys.exit()
    imgdir = file_name[6]

    #process
    img = my_cv_imread(imgdir)
    ele=cv2.mean(img);brigh = (ele[0]+ele[1]+ele[2])/3;print("src:"+str(brigh))
    dst = modifyImgBeta(img,105)
    ele=cv2.mean(dst);brigh = (ele[0]+ele[1]+ele[2])/3;print("dst:"+str(brigh))
    basefile_name = os.path.basename(imgdir)
    cv2.imencode('.jpg', dst)[1].tofile("dst/"+basefile_name)

    dstgray_c = cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(3,(8,8))
    dstgray = clahe.apply(dstgray_c)

    dstgray = cv2.equalizeHist(dstgray_c)
    #dstgray=cv2.GaussianBlur(dstgray,(3,3),0)
    #cv2.imshow('image', dstgray)
    #cv2.waitKey(0)
    dstdf = sobelimg(dstgray)
    #计算每列的平均值
    meanA_row = dstdf.mean(axis=0) 
    sumAll = meanA_row.sum()
    #计算左区域的求和分布
    sumLeft_all = np.zeros(len(meanA_row))
    e_sum=0
    for i in range(len(meanA_row)):
        e_sum = e_sum + meanA_row[i]
        sumLeft_all[i] = e_sum
    #计算右边区域的求和分布
    sumRight_all = np.zeros(len(meanA_row))
    e_sum=0
    for i in range(len(meanA_row)-1,-1,-1):
        e_sum = e_sum + meanA_row[i]
        sumRight_all[i] = e_sum
    #计算左画线与
    rowsize = len(meanA_row)
    mindiff = -1000000;minr=0;minl=0
    for l in range(rowsize//2):
        for r in range(l+20,rowsize-20,1):
            sumM = sumAll-sumRight_all[r]-sumLeft_all[l]
            sumSide = sumAll-sumM
            meanM = sumM/(r-l-1)
            meanSide = sumSide/(rowsize-r+l+1)
            delta = meanSide-meanM
            if delta>mindiff:
                mindiff=delta;minr=r;minl=l

    #求sumLeft_all 差分
    diff_sumLeft_all = np.convolve(meanA_row, np.ones((10,))/10,"same")
    showplt(diff_sumLeft_all)

    minr,minl = getBigDiff(diff_sumLeft_all,50)
    
    #np.set_printoptions(threshold=np.inf)   
    print(minr)
    print(minl)
    # 起点和终点的坐标
    ptStart = (minl, 0)
    ptEnd = (minl, dstgray.shape[0])
    cv2.line(dstgray, ptStart, ptEnd, 255, 2, 4)
    ptStart = (minr, 0)
    ptEnd = (minr, dstgray.shape[0])
    cv2.line(dstgray, ptStart, ptEnd, 255, 2, 4)

    cv2.imshow('image', dstgray)
    cv2.waitKey(0)

    
    