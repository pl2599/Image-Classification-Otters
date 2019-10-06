#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[2]:


def downloadImages(path, folder, file, max_pic_n):
    #This function takes in the path, folder, and txt file names and downloads all the images
    
    dest = path/folder
    dest.mkdir(parents = True, exist_ok = True)
    
    download_images(path/file, dest, max_pics = max_pic_n)


# # Uncomment below to convert to py script

# In[4]:


#!jupyter nbconvert --to script functions.ipynb

