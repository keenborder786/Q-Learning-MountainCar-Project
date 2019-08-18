# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 23:09:46 2019

@author: MMOHTASHIM
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE=10
HM_EPISODES=25000
MOVE_PENALTY=1
ENEMY_PENALTY=300
FOOD_REWARD=25
epsilon=0.9
EPS_DECAY=0.998
SHOW_EVERY=3000


start_q_table=None ##or filename

LEARNING_RATE=0.1
DISCOUNT=0.95

PLAYER_N=1
FOOD_N=2
ENEMY_N=3

d={1:(255,175,0),2:(0,255,0),3:(0,0,255)}##BGR format

class Blob:
    def __init__(self):
        self.x=np.random.randint(0,size)
        self.y=np.random.randint(0,size)
    def __str__(self):
        return f"{self.x},{self.y}"
    def __sub__(self,other):
        return (self.x-other.x,self.y,other.y)
    def action(self,choice):
        if choice==0:
            self.move(x=1,y=1)
        elif choice==1:
            self.move(x=-1,y=-1)
        elif choice==2:
            self.move(x=-1,y=1)
        elif choice==3:
            self.move(x=1,y=-1)
    def move(self,x=False,y=False):
        pass
        
        


