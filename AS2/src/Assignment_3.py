import cv2
import numpy as np
import sys

#Referred from few parts from my Assignment1 code.
def app_description():
    print("Choose an option")
    print("'h': Harris Corner Detection.")
    print("'f': Feature vector calculation for corner detection.")
    print("'l': Localization for corner detection.")
    print("'q': Quit Application")
	
#Referred to my Assignmnet1 code.
def Capture_Image():
    #Get file passed from command line
    if len(sys.argv) == 3:
        filename1 = sys.argv[1]
        filename2 = sys.argv[2]
        image1 = cv2.imread(filename1)
        image2 = cv2.imread(filename2)
  
        #image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        #image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) 
        
        all_images = np.concatenate((image1, image2), axis=1)
        
        cv2.imshow("Image frame", all_images)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    return image1, image2  
    
#Referred from few parts from Assignment1 code.
def gradient_calculator(image, var):
    '''
    image = Input is the original Grayscale Image.
    n = Variance for Calculation of the kernel of the SMoothened image. 
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image)
    kernel = np.ones((var,var),np.float32)/(var*var)
    dst=  cv2.filter2D(image,-1,kernel)
    dy, dx = np.gradient(dst)
    
    return dy, dx

    
def harris(image, var, k, thresh, n):
    '''
    var= Variance of Gaussian which decides how much the image should be blurred.
    k = coefient of trance usually decides if the detected shape is edge or corner.(typically user selected and in the range of 0 to 0.5 for corner detection)
    threshold = Threshold to detect corner if this value is smaller than the calculated response(R). 
    n = Neighborhood(Number of pixels to accounted for which calculating the corners)
    '''
    print(var, k, thresh, n)

    #Find the height and weight of the image to iterate over the image. 
    height = image.shape[0]
    weight = image.shape[1]
        
    # Step 1 : Compute the derivate ox and y of an image 
    dx, dy = gradient_calculator(image, var) 
  
    #Step 2: Compute product of Derivative at each pixel. 
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2  
    
    #Step 3: Compute the sum of the product of the derivate at each pixel.
    
    #Create a dummy copy for plotting the corners on the image. 
    dummy = image.copy()

    window = int(n/2)
    
    for y in range(window, height - window):
        for x in range(window, weight - window):
            Sx1 = Ixx[y - window : y + window + 1, x - window : x + window + 1]
            Sy1 = Iyy[y - window : y + window + 1, x - window : x + window + 1]
            Sxy1 = Ixy[y - window : y + window + 1, x - window : x + window + 1]
            
            Sx2 = Sy1.sum()
            Sxy2 = Sxy1.sum()
            Sy2 = Sy1.sum()

            #Hence, the variables Sx2, Sxy, Sy2 are the components of the Correlation matrix,
            #whose derivative and trace will be used to calculate the response as follows. 
            det = (Sx2 * Sy2) - (Sxy2 ** 2)
            trace = Sx2 + Sy2 
            Response =  det - k *(trace ** 2) 
            
            #Compare this response with the threshold. If Response > Tou (threshold) then corner is detected.

            if Response > thresh:
                dummy.itemset((y, x, 0), 0)
                dummy.itemset((y, x, 1), 0)
                dummy.itemset((y, x, 2), 255)
                #Use Green coloured rectangles to highlight the corners in the image.
                cv2.rectangle(dummy, (x + 10, y + 10), (x - 10, y - 10), (0, 255, 0), 1)
                     
                
    #cv2.namedWindow("Harris Corners detected in Blue.", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Output", dummy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Referred OFFICIAL OPENCV LIBRARY TUTORIAL page:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html 
def localization(image1, image2):
    
    img = np.concatenate((image1, image2), axis=1)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_scale = np.float32(gray_scale)
    dst = cv2.cornerHarris(gray_scale,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray_scale,np.float32(centroids),(5,5),(-1,-1),criteria)
    output = np.hstack((centroids,corners))
    output = np.int0(output)
    img[output[:,1],output[:,0]]=[255,0,0]
    img[output[:,3],output[:,2]]=[0,255,0] 
    
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     
#Referred Brute-force feature matching tutorial from OFFICCIAL OPENCV Library link:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#basics-of-brute-force-matcher 

def Feature_Vector(image1, image2):

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1,None)
    kp2, des2 = orb.detectAndCompute(image2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    #Define empty lists to collect the points to be numbered
    ls1 = []
    ls2 = []
    for m in matches:
        ls1.append((kp1[m.queryIdx].pt))
        ls2.append((kp2[m.trainIdx].pt))
    for p in range(0, 80):
        p1 = ls1[p]
        p2 = ls2[p]
        #Number the matching points, put it as text in the image and mark it in Blue color.
        cv2.putText(image1, str(p), (int(p1[0]), int(p1[1])),  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.putText(image2, str(p), (int(p2[0]), int(p2[1])),  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    output = np.concatenate((image1, image2), axis=1)
    
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#Referred to my Assignmnet1 code.
def main():
    print("Inside main")
    
    #Displays operation options on the console.
    app_description()
    
    #Capture the image from console or from desktop camera.
    image1, image2 = Capture_Image()  
    
    k = input()
    
    while k != 'q':
        if k == 'h':
            var =  int(input("Enter the variance for Gaussian Scale(typically any range between 0 to 5: "))
            k = float(input("Enter the coeeficient of trace parameter(typically any range between 0 to 0.5: "))
            thresh = int(input("Enter the threshold(typically any range between 100000 to 700000): "))
            n = int(input("Enter the neighborhood you want to include in calucation to detect the corner(typically between 0 to 5): "))
            harris(image1, var, k, thresh, n)
        if k == 'f':
            Feature_Vector(image1, image2)
        if k == "l":
            localization(image1, image2)  
        k = input()



if __name__ == '__main__':
	main() 