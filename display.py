import csv
import cv2 as cv
import numpy as np
import os

data = []
kernel = np.ones((19,19 ), np.uint8)
# test.append(["image_name","x1","x2","y1","y2"])
with open('testnew.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = list(csv_reader)
csv_file.close()
    # line_count = 0
    # # im_count = 0
    # for row in csv_reader:
    #     if line_count == 0:
    #         print(f'Column names are {", ".join(row)}')
    #         line_count += 1
    #     # elif line_count == 15:  
    #     #     break
    #     else :
            # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
def my_function(img):
    # img2 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # h,w = img2.shape
    # w = int(h/500)
    # h = int(w/500)
    blur = cv.blur(img, (5,5), 0)
    
    gradient = cv.morphologyEx(~blur, cv.MORPH_GRADIENT, kernel)
    ret2,th2 = cv.threshold(gradient,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)  
    blur = cv.GaussianBlur(th2, (5,5), 0)
    ret2,th2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # kernel = np.ones((3,3 ), np.uint8)
    # gradient = cv.morphologyEx(th2, cv.MORPH_GRADIENT, kernel)
    # blur = cv.GaussianBlur(gradient, (5,5), 0)
    # ret2,th2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # blur = cv.GaussianBlur(gradient, (5,5), 0)
    # imagem = cv.bitwise_not(th2)
    image, contours, hierarchy = cv.findContours(blur,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    # print(len(contours) )
    max_area = 0
    # max_2 = 0
    # max_2i = 0
    max_index = 0
    for i, cnt in enumerate(contours):
        if cv.contourArea(cnt) > max_area:
            # max_2i = max_index
            max_index = i
            # max_2 = max_area
            max_area = cv.contourArea(cnt)
            # print(max_index)
    # if max_2 == 0:
    #     max_2i = max_index
    x,y,w,h =  cv.boundingRect(contours[max_index])
    # test.append([name,

    # cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    # cv.imshow("image", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return x,    x+w,    y,    y+h
    # ])

# cv.rectangle(img,(int(row[1]), int(row[3])),(int(row[2]), int(row[4])),(0,255,0),3)




# line_count += 1
# im_count += 1
# elif line_count == 10000:
#     break
# else:
#     line_count += 1
# print(f'Processed {im_count} images.')

# img = cv.imread('images/', 0)
# folder = "img"
# for filename in os.listdir(folder):
#         img = cv.imread(os.path.join(folder,filename), 0)
#         if img is not None:
#             my_function(filename, img)
# print(int(len(data)/1200))
for i in range(1, len(data)):
    print(i)
    img = cv.imread("images/"+data[i][0], 0)
    data[i][1] , data[i][2], data[i][3], data[i][4] = my_function(img)
    # cv.rectangle(img, (data[i][1],data[i][3]), (data[i][2],data[i][4]),(0,0,255),3)
    # cv.rectangle(img,(kachara[0],kachara[0]+kachara[1]),(kachara[1],kachara[1]+kachara[3]),(255,0,0),3)
    # mask = np.zeros(img.shape[:2], np.uint8)
    # backgroundModel = np.zeros((1, 65), np.float64) 
    # foregroundModel = np.zeros((1, 65), np.float64) 
    # cv.imshow(data[i][0], img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

with open('testnew1.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(data)

csvFile.close()