'''
Created by Xiaofei Huang (xhuang@ece.neu.edu)
'''
import numpy as np
import os
import cv2
from opendr.simple import *
from opendr.renderer import ColoredRenderer
from opendr.renderer import TexturedRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smil_webuser.serialization import load_model
import random
from pickle import load

## Only needed for pose prior
class Mahalanobis(object):

    def __init__(self, mean, prec, prefix):
        self.mean = mean
        self.prec = prec
        self.prefix = prefix

    def __call__(self, pose):
        if len(pose.shape) == 1:
            return (pose[self.prefix:]-self.mean).reshape(1, -1).dot(self.prec)
        else:
            return (pose[:, self.prefix:]-self.mean).dot(self.prec)

syn_folder = '/home/faye/Documents/smil/outputs'

## Assign attributes to renderer
w, h = (640, 480)

## Load SMIL model
m, kin_table = load_model('smil_web.pkl')
tmpl = load_mesh('template.obj')


## List background images
bg_folder = '/home/faye/Documents/smil/bg_img'
bg_list = []                                                                                                            
bg_subdirs = [x[0] for x in os.walk(bg_folder)]                                                                            
for subdir in bg_subdirs:                                                                                            
    files = os.walk(subdir).__next__()[2]                                                                             
    if (len(files) > 0):                                                                                         
        for file in files:                                                                                        
            bg_list.append(subdir + "/" + file)       
#print(bg_list)


## List texture images
txt_folder = '/home/faye/Documents/smil/textures'
txt_list = []                                                                                                            
txt_subdirs = [x[0] for x in os.walk(txt_folder)]                                                                            
for subdir in txt_subdirs:                                                                                            
    files = os.walk(subdir).__next__()[2]                                                                             
    if (len(files) > 0):                                                                                          
        for file in files:                                                                                        
            txt_list.append(subdir + "/" + file)        
#print(txt_list)

num = 0
bodies_folder = '/home/faye/Documents/smil/bodies'
for x in os.walk(bodies_folder):  
    if x[0] == bodies_folder:
        continue 
                                                                                         
    cur_body_file = os.path.join(x[0], '000.pkl')
    cur_conf_file = os.path.join(x[0], 'conf.yaml')

    body_params = load(open(cur_body_file, 'rb'))
    #print(body_params)

    m.pose[:3] = body_params['global_orient']
    m.pose[3:] = body_params['body_pose']
    m.betas[:] = body_params['betas']
    trans = body_params['camera_translation'][0]

    g_rot0 = float(m.pose[0])
    g_rot1 = float(m.pose[1])          
    g_rot2 = float(m.pose[2])

    for i in range(10):
        num = num + 1
        # syn: change global rotation
        # m.pose[0] = g_rot0 - np.pi/18 * (i)
        # m.pose[1] = g_rot1 + np.pi/18 * (i)
        # m.pose[2] = g_rot2 + np.pi/18 * (i)

        m.pose[0] = g_rot0 + np.pi/11 * (i)
        m.pose[1] = g_rot1 - np.pi/11 * (i)
        m.pose[2] = g_rot2 - np.pi/11 * (i)


        bg_idx = num % len(bg_list)
        bg_file = bg_list[bg_idx]
        bg = cv2.imread(bg_file)
        x = 0
        y = 0
        bg_im = bg[y:y+h, x:x+w].astype(np.float64)/255
      
        txt_idx = num % len(txt_list)
        txt_file = txt_list[txt_idx]
        txt = cv2.imread(txt_file)
        txt = cv2.cvtColor(txt, cv2.COLOR_BGR2RGB)
        txt_im = txt.astype(np.float64)/255 

        cam = ProjectPoints(v=m.r, rt=np.zeros(3), t = np.array([0, -.1, .5]), f=np.array([w,w])/2., c=np.array([w,h])/2, k=np.zeros(5))

        rn = TexturedRenderer(v=m.r, f=m.f, vc=np.ones_like(m.r), vt=tmpl.vt, ft=tmpl.ft,
                                 texture_image = txt_im, background_image = bg_im,
                                 camera = cam,
                                 frustum = {'near': .1, 'far': 10., 'width': w, 'height': h},
                                 overdraw=False)
        data = 255 * rn.r
        cv2.imshow('render_SMIL', rn.r)
        
        file_name = 'syn' + str(num) + '.jpg'
        syn_file = os.path.join(syn_folder, file_name)
        cv2.imwrite(syn_file, data)


        

    





