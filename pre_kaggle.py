import cv2 , glob , numpy , pdb
def scalRadius(img,scale):
    x = img[int(img.shape[0]/2),:,:].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)
scale = 300
# pdb.set_trace()
for f in glob.glob("/media/dani/0658C3F958C3E591/Kaggle_diabetic/realdataset/dataset/train/1/*.jpeg")+glob.glob("/media/dani/0658C3F958C3E591/Kaggle_diabetic/realdataset/dataset/train/2/*.jpeg"):
    try:
        a = cv2.imread(f)
        a = scalRadius(a,scale)
        a = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)
        b = numpy.zeros(a.shape)
        cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
        a = a*b+128*(1-b)
        aa = cv2.resize(a,(512,512))
        cv2.imwrite(f,aa)
    except:
        print(f)
