import cv2
import numpy as np
import pickle
from imutils.perspective import order_points
from keras.models import load_model
from skimage.segmentation import clear_border
from keras.preprocessing.image import img_to_array
import solver as sv
import time

# load the trained convolutional neural network and the label
# binarizer
start_time = time.time()

print("[INFO] Loading Model...")
model = load_model("Soduku.model")
lb = pickle.loads(open("Label.pickle", "rb").read())

start_time2 = time.time()

def recognizer(frame,filename):
    
    #rgb = cv2.resize(frame,(int(frame.shape[0]*0.5),int(frame.shape[1]*0.5)), interpolation = cv2.INTER_AREA)
    rgb = cv2.resize(frame,(800,600), interpolation = cv2.INTER_AREA)

    #convert image to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    #blur the image
    blurred = cv2.GaussianBlur(gray,(5,5),0)

    #apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    _,cnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #sorting the 10 largest contours with highest area first
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:10]

    src = None
    print("[INFO] Framing Sudoku")
    for c in cnts:
        #approximate the contours
        peri = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, 0.02*peri ,closed=True)

        #only if a valid square region
        if len(approx) == 4:
            src = approx
            break

    cv2.drawContours(rgb,[src],-1,(0,0,255),2)

    pts = src.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
     
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
     
    # multiply the rectangle by the original ratio
    #rect *= ratio

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
     
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
     
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
     
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect,dst)
    warp = cv2.warpPerspective(rgb,M,(maxWidth,maxHeight))
    cv2.imshow('warp',warp)
    cv2.waitKey(0)


    #convert to gray and find the sliding window width & height
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    winX = int(warp.shape[1]/9.0)
    winY = int(warp.shape[0]/9.0)

    #empty lists to hold recognized digits and center co-ordinates of the cells
    labels = []
    centers = []
    print("[INFO] Initializing Digit Recognition")

    #slide the window through the puzzle
    for y in range(0, warp.shape[0], winY):
        for x in range(0, warp.shape[1], winX):

            #slice the cell
            window = warp[y:y+winY,x:x+winX]

            #sanity check
            if window.shape[0] != winY or window.shape[1] != winX:
                continue

            #clone warp image to draw windows in the end
            clone = warp.copy()

            digit = cv2.resize(window,(100,100))
            _,digit = cv2.threshold(digit,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
            
            #clear borders
            digit = clear_border(digit)

            #whether an empty cell or not
            numPixels = cv2.countNonZero(digit)
            if numPixels<100:
                label = 0
            else:
                digit = cv2.cvtColor(digit,cv2.COLOR_GRAY2BGR)
                digit = digit.astype("float") / 255.0
                digit = img_to_array(digit)
                digit = np.expand_dims(digit, axis=0)

                # classify the input image
                proba = model.predict(digit)[0]
                idx = np.argmax(proba)
                label = lb.classes_[idx]

            labels.append(int(label))
            centers.append(((x+x+winX-25)//2,(y+y+winY+15)//2))

            #draw rectangle for each cell on warp
            cv2.rectangle(clone,(x,y),(x+winX,y+winY),(0,0,255),2)


    print("[INFO] Finished Digit Recognition")
    start_time3 = time.time()

    row = 9
    col = 9
    sudoku = [labels[col*i : col*(i+1)] for i in range(row)]
    print(sudoku)
        
    #convert to numpy array of 9x9
    grid = np.array(labels).reshape(9,9)

    #find the indices of empty cells
    gz_indices = zip(*np.where(grid==0))

    #center co-ordinates of all the cells
    gz_centers = np.array(centers).reshape(9,9,2)

    if sv.solve(sudoku) == 81:
        print("Done")
        end_time = time.time()
        sv.printBoard(sudoku)
        grid = sudoku
        for row,col in gz_indices:
            cv2.putText(warp,str(grid[row][col]),tuple(gz_centers[row][col]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

        #process the src and dst points
        pt_src = [[0,0],[warp.shape[1],0],[warp.shape[1],warp.shape[0]],[0,warp.shape[0]]]
        pt_src = np.array(pt_src,dtype="float")
        pt_dst = src.reshape(4,2)
        pt_dst = pt_dst.astype("float")

        #align points in order
        pt_src = order_points(pt_src)
        pt_dst = order_points(pt_dst)

        #calculate homography matrix
        H,_ = cv2.findHomography(pt_src,pt_dst)

        #reproject the puzzle to original image
        im_out = cv2.warpPerspective(warp,H,dsize=(gray.shape[1],gray.shape[0]))
        im_out = cv2.addWeighted(gray,0.7,im_out,0.3,0)

        cv2.imshow("Projected",im_out)
        cv2.waitKey(0)
    else:
        print("Error")
    

filename = "3.jpg"
frame = cv2.imread(filename)
recognizer(frame,filename)

cv2.destroyAllWindows()
