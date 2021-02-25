import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import sys


#App Description:
def app_description():
	print("'i': Reload the original image.")
	print("'w': Save the current image in to 'out.jpg'.")
	print("'g': Convert the image to grayscale using the openCV conversion function.")
	print("'G': Conver the image to grayscale using my implementation.")
	print("'c': Cycle through the color channels(press 'q' to break).")
	print("'s': Convert the image to grayscale and smooth using the openCV function.")
	print("'S': Convert the image to grayscale and smooth using user created function.")
	print("'d': Downsample the image by a factor of 2 without smoothing.")
	print("'D': Downsample the image by a factor of 2 with smoothing.")
	print("'x': Convert the image to grayscale and perform  convolution with a x derivative filter.")
	print("'y': Convert the image to grayscale and perform  convolution with a y derivative filter.")
	print("'m': Show the magnitude of the gradient normalized to the range [0, 255].")
	print("'p': Convert the image to grayscale and plot the gradient vector of the image every N pixels\
and let the plotted gradient vector have a length of K.")
	print("'r': convert the image to grayscale and rotate it.\n")
	print("Choose an option(press 'q' to quit):") 
    
def Capture_Image():
    #Get file passed from command line
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        image_frame = cv2.imread(filename)

    #Get video from desktop camera and chop the image into image frames
    elif len(sys.argv) < 2:
        Video = cv2.VideoCapture(0)
        for i in range(0,10):
            value,image_frame = Video.read()
        if value:
            cv2.imwrite("capture.jpg", image_frame) 
            #See the captured Sample Image:
            
    #Get the shape of the image
    print("Image size is:")
    print(image_frame.shape)
    cv2.imshow("Image frame", image_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return image_frame 

def display_captured_image(image_frame):
    cv2.imshow("Image frame", image_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    
def reload_image():
    cv2.destroyAllWindows()
    image = Capture_Image()
    display_captured_image(image)
    
    
def save_image(image):
	cv2.imwrite("output.jpg", image)   
       
def cvt2Gray_opencv(image):
	black_and_white_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return black_and_white_image   
    
def cvt2Gray_usercode(image):
        grey_image = np.average(image, weights=[0.114, 0.587, 0.299], axis=2).astype(np.uint8)
        print(grey_image)
        cv2.imshow('User_implemented_Grayscale',grey_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def cycleColor(k, image):

    #Cycling through the color channels on pressing key
    print("Please press 'c' two more consecutive times to see red and green image Or press ENTER other key to get out of the loop")
    if k == "c":
        blue_image=np.copy(image)
        blue_image[:,:,1:]=0
        cv2.imshow('Blue Channel',blue_image)
        cv2.waitKey(0)
        key = input()
        if(key == "c"):
            green_image=np.copy(image)
            green_image[:,:,(0,2)]=0
            cv2.imshow('Green Channel',green_image)
            cv2.waitKey(0)
            key = input()
            if(key == "c"):
                red_image=np.copy(image)
                red_image[:,:,:2]=0
                cv2.imshow('Red Channel',red_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                print("Getting out of the color cycle loop, Please press the another operation again")
        else:
            print("Getting out of the color cycle loop, Please press the another operation again")

def smooth(image):
	n = 7
	kernel = np.ones((n, n), np.float32)/(n * n)
	output = cv2.filter2D(image, -1, kernel)
	return output 

def smooth_opencv(image):

    global grey_image
    grey_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    def sliderHandler(n): 
        if n ==0:
            kernel=np.ones((1,1),np.float32)/(1*1)
            dst=cv2.filter2D(grey_image,0,kernel)
            cv2.imshow('processed',dst)
        else:
            kernel=np.ones((n,n),np.float32)/(n*n)
            dst=cv2.filter2D(grey_image,-1,kernel)
            cv2.imshow('processed',dst)
        
    cv2.imshow('blur',grey_image)
    #(trackbarName, windowName, value, count, onChange) 
    cv2.createTrackbar('s','blur',0,15,sliderHandler)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
        
def smooth_usercode(image):
    global grey_image_new
    grey_image_new = np.average(image, weights=[0.114, 0.587, 0.299], axis=2).astype(np.uint8)	
    
    def sliderHandler_usercode(n):
        blurred = gaussian_filter(grey_image_new, sigma = n)
        cv2.imshow("Image", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
	
    cv2.imshow("Display window", grey_image_new)
    cv2.createTrackbar("S", "Display window", 0, 10, sliderHandler_usercode)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    
def downSample_opencv(image):
    ds = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
    print("Downsampled image size: ", ds.shape)
    cv2.imwrite("downsample1.jpg", ds)
    cv2.imshow("Display window", ds)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def downSample_usercode(image):
    smoothened_image = smooth(image)
    ds = cv2.resize(image, (int(smoothened_image.shape[1] / 4), int(smoothened_image.shape[0] / 4)))
    print("Downsampled image size: ", ds.shape)
    cv2.imwrite("downsample2.jpg", ds)
    cv2.imshow("Display window", ds)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def xdrv(image):
    grey_image = cvt2Gray_opencv(image)
    smoothen_image = smooth(grey_image)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(grey_image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    normalized_grad_x = cv2.normalize(grad_x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow("Display window", normalized_grad_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ydrv(image):
    grey_image = cvt2Gray_opencv(image)
    smoothen_image = smooth(grey_image)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_y = cv2.Sobel(grey_image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    normalized_grad_y = cv2.normalize(grad_y, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow("Display window", normalized_grad_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gradientVector(image):
    grey_image = cvt2Gray_opencv(image)
    smoothen_image = smooth(grey_image)
    dX = cv2.Sobel(grey_image, cv2.CV_32F, 1, 0, (3,3))
    dY = cv2.Sobel(grey_image, cv2.CV_32F, 0, 1, (3,3))
    mag, direction = cv2.cartToPolar(dX, dY, angleInDegrees=True)
    abs_mag = cv2.normalize(mag, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow("Display window", abs_mag)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

#Reference for code: https://stackoverflow.com/questions/32907838/plotting-a-gradient-vector-field-in-opencv
def plot_gradient(image):
    grey_image = cvt2Gray_opencv(image)
    grad_vec = 10

    def gradient_slider(pixels):
        if pixels == 0:
            cv2.imshow("Image", grey_image)
            return
        rows, cols = grey_image.shape[:2] 
        xgrad, ygrad = numpy.gradient(grey_image)
        mag = numpy.sqrt(xgrad * xgrad + ygrad * ygrad)

        row = 0
        col = 0
        while row < rows:
            col = 0
            while col < cols:
                y_grad = ygrad[row][col]
                x_grad = xgrad[row][col]

                if x_grad:
                    angle = math.atan(y_grad / x_grad)
                else:
                    angle = 1.5708 
                magnitude = mag[row][col] 
                if magnitude:
                    x = magnitude * math.cos(angle)
                    y = magnitude * math.sin(angle)

                    x *= grad_vec / magnitude
                    y *= grad_vec / magnitude

                    x += row
                    y += col
                    cv2.arrowedLine(grey_image, (col, row), (int(y), int(x)), (0, 0, 255), 2)
                col += pixels
            row += pixels
        cv2.imshow("Image", grey_image) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.createTrackbar("G", 'Gradient', 0, 200, gradient_slider)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotation(image):
    global grey_image
    grey_image = cvt2Gray_opencv(image)
    
    def rotation_slider(degree):
        cols = grey_image.shape[0]
        rows = grey_image.shape[1]
        M = cv2.getRotationMatrix2D((rows/2, cols/2), degree, 1)
        dst = cv2.warpAffine(grey_image, M, (rows, cols))
        cv2.imshow("Display window", dst)
        
    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Rotate", "Display window", 0, 360, rotation_slider)
    cv2.imshow("Display window", grey_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def main():
    print("Inside main")
    
    #Displays operation options on the console.
    app_description()
    
    #Capture the image from console or from desktop camera.
    image = Capture_Image()  
    
    img = image.copy()
    
    key_dict = {"g": cvt2Gray_opencv(img)}
    
    k = input()
    while k != 'q':
        if k == 'i':
            reload_image()
        elif k == 'w':
            save_image(image)
        elif k == "c":
            cycleColor(k, image)
        elif k == "g":
            display_captured_image(cvt2Gray_opencv(image))
        elif k == "G":
            cvt2Gray_usercode(image)
        elif k == 's':
            smooth_opencv(image)
        elif k == 'S':
            smooth_usercode(image)
        elif k == 'd':
            downSample_opencv(image)
        elif k == 'D':
            downSample_usercode(image)
        elif k == 'x':
            xdrv(image)
        elif k == 'y':
            ydrv(image)
        elif k == 'm':
            gradientVector(image)
        elif k == 'p':
            plot_gradient(image)
        elif k == 'r':
            rotation(image)
            
        k = input()
    

if __name__ == '__main__':
	main() 

