import numpy as np 
from scipy.misc import imread, imsave, imresize

def load_servo():

	""" load and preprocess the images """
	X=np.zeros((800,3,32,32))
	y=np.zeros((800,2))
	n=0
	for i in xrange(1,201):
		for j in xrange(1,5):
			vservo4_dir="Dataset/rgb"+str(i)+str(j)+".jpg"
			img=imread(vservo4_dir)
			X[n]=imresize(img,(32,32)).transpose(2,0,1)
			n+=1
    
	y[0:800:4]=[5,2]
	y[1:800:4]=[3,2]
	y[2:800:4]=[2,3]
	y[3:800:4]=[0,0]
	X_train=X[0:700]
	y_train=y[0:700]
	X_val=X[700:800]
	y_val=y[700:800]


	#preprocess the images
	mean_image=np.mean(X_train,axis=0)
	X_train-=mean_image
	X_val-=mean_image
	X_test,y_test=X_val,y_val

	return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test }
		









