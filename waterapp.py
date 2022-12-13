# using streamlit to build a real-time demo of YOLO model
import streamlit as st
import cv2
import pandas as pd
import os,glob
import numpy as np
import torch
#import time ,sys
#from streamlit_embedcode import github_gist
#import urllib.request
#import urllib
#import moviepy.editor as moviepy
import copy

from fastai.vision.core import *
from fastai.vision.data import *
from fastai.vision.all import *
from fastai.vision import *
from fastai.vision.core import PILImage, PILMask
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
from shapely.geometry import Polygon

waterway_model_file='model_version_1.pth'
learn_inf = load_learner(waterway_model_file)
save_path='/image/'
classes=learn_inf.dls.train.after_item.vocab
scale=round((5.36+6.65+7.77)/3)
unit=scale**2/10000


def get_label_image(fpath,imgfile):
  label_file=fpath.joinpath(str(imgfile.name).replace('.JPG','_lb.png'))
  return label_file

def get_y(img_file):
  maskfile=get_label_image(path_lab,img_file)
  return get_msk(maskfile)

def acc_camvid(inp, targ):
  targ = targ.squeeze(1)
  mask = targ != void_code
  return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

def apply_mask(image, mask, color,num, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask[:,:,c] == num,image[:, :, c] *(1 - alpha) + alpha * color[c] * 255,image[:, :, c])
    return image

def num2id(imgvalue):
    # value is based on processing method
    classes=["background","pond","waterway","wgrass","eground"]
    color_code=[(0.0, 0.0, 0.0),(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),(1.0, 1.0, 1.0)]
    ind=[0,1,2,3,4]
    ind=[ x*60 for x in ind]
    try:
        clsid=ind.index(imgvalue)
        clsname=classes[clsid]
        color=color_code[clsid]
        return [clsid,clsname,color]
    except ValueError:
        pass

def maskimage_to_apply(fraw,fmask):
    #fraw: the original image to detect/ image data
    #fmask: output image with mask/ image data    
    img=np.array(fraw)
    masked_image = img.copy()
    maskarr = np.array(fmask)
    indx=np.unique(fmask) # skip order 0 & last one
    #print(indx)
    idls=[num2id(x) for x in indx]
    #print(idls)
    masknum=len(indx)-1    
    maskinfo={}
    maskname=[]
    for i in range(masknum):
        maskname.append(idls[i+1])
        masked_image = apply_mask(masked_image, maskarr, idls[i+1][2],indx[i+1])
    maskinfo['classname']=maskname
    maskinfo['data']=masked_image
    return maskinfo#masked_image 

def mask_to_polygon(maskfile):
    maskarr = maskfile #np.array(msk)
    contours, hierarchy= cv2.findContours(maskarr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    area=[]
    polygons=[]
    #print(len(contours))
    for poly in contours:
        #print(cv2.contourArea(poly))
        s=poly.tolist()
        s=[ x[0]  for x in s]
        polygon = Polygon(s)
        area.append(polygon.area)
        polygons.append(polygon)
    return polygons        



def object_detection_image():
    st.title('Waterway Detection for Images')
    st.subheader("""
    This app will detect the waterway in an image and outputs the image with polygons.
    """)
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    if file!= None:
        st.write("Image Uploaded Successfully:")
        img=PILImage.create(file)
        
        st.image(img, caption = "Uploaded Image")
        my_bar = st.progress(0)
        #confThreshold =st.slider('Confidence', 0, 100, 50)
        #nmsThreshold= st.slider('Threshold', 0, 100, 20)

        pred = learn_inf.predict(img)
        test=pred[0].numpy()
        test[test == 1]=60
        test[test == 2]=60*2
        test[test == 3]=60*3
        test[test == 4]=60*4
        #pred_arg=test
        ind=np.unique(test)
        if len(ind)>=2:
          #rescaled = (255.0 / pred_arg.max() * (pred_arg - pred_arg.min())).astype(np.uint8)
          rescaled=test
          maskimg = Image.fromarray(rescaled.astype(np.uint8))
          szm=maskimg.shape
          img=img.resize((szm[1],szm[0]))
          #test=np.array(Image.fromarray(rescaled).resize((sz[1],sz[0])))
          stacked_img = np.stack((rescaled,)*3, axis=-1)
          maskinfo=maskimage_to_apply(img,stacked_img)
          #print(maskinfo['classname'])
          #plt.imshow(maskinfo['data'])
          img2 = Image.fromarray(maskinfo['data'])
          draw = ImageDraw.Draw(img2)
          obj_list=[ x[1] for x in maskinfo['classname']]
          polys=mask_to_polygon(rescaled.astype(np.uint8))
          area=[]
          for x in polys:
              cx = x.representative_point().x
              cy = x.representative_point().y
              draw.text((cx,cy), '{:8.1f}'.format(x.area*unit), stroke_fill=(255, 0, 0),fill=255,font=font)
              area.append(x.area)
            img2.save(save_path/'imagesave.png')
        else:
          print('no results')
        df= pd.DataFrame(list(zip(obj_list,area)),columns=['Object Name','Area'])
        #df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
        
        if st.checkbox("Show Object's list" ): 
            st.write(df)
#        if st.checkbox("Show Confidence bar chart" ):
#            st.subheader('Bar chart for confidence levels')
#            st.bar_chart(df["Area"]) 
            
        st.image(img2, caption='Proccesed Image.')
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        my_bar.progress(100)



def main():
#    new_title = '<p style="font-size: 42px;">Welcome to water wat detection demo </p>'
#    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

#    read_me = st.markdown("""
#    This Web app is for image model demo and test usage!"""
#    )
#    st.sidebar.title("Select Activity")
#    choice  = st.sidebar.selectbox("MODE",("About","Detection(Image)","(Coming soon) Detection(Video)"))
       
    if st.button('Classify'):
        object_detection_image()
        #pred, pred_idx, probs = learn_inf.predict(img)
        #st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
    else: 
        st.write(f'Click the button to classify') 
'''    
    if choice == "Object Detection(Image)":
        #st.subheader("Object Detection")
        read_me_0.empty()
        read_me.empty()
        #st.title('Object Detection')
        object_detection_image()
    elif choice == "Object Detection(Video)":
        read_me_0.empty()
        read_me.empty()
        #object_detection_video.has_beenCalled = False
        object_detection_video()
        #if object_detection_video.has_beenCalled:
        try:

            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4','rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video") 
        except OSError:
            ''

    elif choice == "About":
        print()
'''        

if __name__ == '__main__':
		main()	