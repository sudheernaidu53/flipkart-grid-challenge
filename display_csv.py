import csv
import cv2 as cv

# F:\ml\machine-learning-ex1\
with open('test2.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = list(csv_reader)
csv_file.close()

for i in range(1, int(len(data)/40)):
    img = cv.imread("images/"+data[i][0], 1)
    cv.rectangle(img,(int(data[i][1]), int(data[i][3])),(int(data[i][2]), int(data[i][4])),(0,255,0),3)
    cv.imshow(data[i][0], img)
    cv.waitKey(0)
    cv.destroyAllWindows()