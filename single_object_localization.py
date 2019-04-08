
# coding: utf-8

# ## Evolution of Object Detection and Localization Algorithms 
# ### Code implementation on Pascal VOC dataset and ResNet for transfer learning

# The codes used in this notebook is executing the ideas presented in below shared blog. Almost all of the content in this notebook is taken from **Jeremy Howard's fast.ai** deep learning part 2 course. The blog was motivated from **Andrew Ng's deeplearning.ai** CNN course. I would recommend you to go through below blogpost to develop the intuitive understanding before diving into the codes.
# 
# https://towardsdatascience.com/evolution-of-object-detection-and-localization-algorithms-e241021d8bad

# ### Data Preprocessing

# #### Libraries used

# In[206]:


#! ln -s ../fastai/fastai/ .


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Install `fastai` library by following instructions mentioned in below link
# 
# https://github.com/fastai/fastai

# In[7]:


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from IPython.display import Image
from matplotlib import patches, patheffects


# In[209]:


# check to make sure you set the device
#torch.cuda.set_device(0)  # set.device(0) to work on CPU


# #### Understanding Pascal VOC data
# http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

# First of all, download the Pascal VOC data by running below 2 lines of codes. After running, I commented it out

# In[210]:


#!wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar


# In[211]:


#!wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip


# In[212]:


# Use to move image data from 2nd link to PASCAL_VOC from 1st link
# ! mv pascal_data/VOCdevkit/VOC2007 pascal_data/PASCAL_VOC/


# Let's try to visualize the training data, which contains data about **class** of an image and **bounding box** related data in each image. Bounding box data contains **height, width, top left coordinate, top right coordinate** around each object in image

# In[10]:


get_ipython().system(' ls pascal_data/PASCAL_VOC/  # json (bbox, class, image id)')


# In[11]:


PATH = Path('pascal_data/PASCAL_VOC/')


# In[12]:


list(PATH.iterdir())


# I am going to use training data from `pascal_train2007.json` as I am working on CPU and it has relatively small amount of data. If anyone is working on GPU, feel free to use 2012 data also. Let's see how it looks like

# In[13]:


train_json = json.load((PATH/'pascal_train2007.json').open()) #  PATH.open() creates a io.text wrapper that is like file path


# In[14]:


train_json.keys()


# What's there in each of those keys. Let's see one by one

# In[217]:


train_json['images'][:2]


# `images` has data about `image ids` and `image files`. Let's see where are these images. e.g. `000012.jpg`. 
# (They are present in `VOC2007/` as can be seen in above ls statement)

# Let's just move `VOC2007` into `PASCAL_VOC` to make it accessible via PATH iterator

# In[218]:


# ! mv pascal_data/VOCdevkit/VOC2007 pascal_data/PASCAL_VOC/


# In[15]:


JPEGS = 'VOC2017/JPEGImages'


# In[16]:


IMAGE_PATH = PATH/JPEGS


# In[17]:


list(IMAGE_PATH.iterdir())[:4]  


# So, here are our jpg images. 
# 
# Let's now look at other 3 keys of `train_json`

# In[18]:


train_json['type'] # doesn't interest me


# In[19]:


train_json['annotations'][:2]


# Ok. So, it has `bbox` data, `id` of image to map with `jpg` using `images` key. `category_id` as final label. Categori ID has name in `categories` key. Let's take a look

# In[20]:


train_json['categories']


# It has only **20 unique** categories in data

# How much is total training data?

# In[21]:


len(list(IMAGE_PATH.iterdir()))


# #### Visualizing images with bounding boxes

#  The bbox field specifies the bounding box of the object in the image, as `[left,top,right,bottom]`

# In[22]:


def show_img(im, figsize=None, ax=None):
    """
    Wrapper to make use of subplots to show images. (no need x axis, y axis lines when we are showing images)
    
    Input:
    im: input image that we want to show
    
    Returns:
    ax: matplotlib axis element that can be used later if we want to change linewidth/text in that
    """
    
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


# In[23]:


def draw_outline(o, lw):
    """
    Draws outline around text/ bounding box
    
    To make text/ bounding boxes visible in any image, we use thick white colored bounding box with black outline.
    
    Input:
    o: bbox patch or text box
    lw: Float. Thickness of text/ bbox
    """
    
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


# In[24]:


Image('image_bbox.png')


# * In below function, we are switching the x,y coordinates to y,x. Why?
# Because in computer vision world, when we say 640x480, it means width by height of screen. 
# But in math world, when we talk about array, 640, 480 would be rows, columns which translates to height, width. 
# 
# So we are changing format of computer vision world to math/numpy world to make it easier for to think of dimensions later. 
# 
# As quoted by Jeremy here : https://youtu.be/b8D6Bwck9QM?t=3596

# In[25]:


def bb_hw(b):
    """
    Return bbox in y,x, height, width format. 
   
    Input: 
    b: bbox are given from pascal voc dataset
    
    Returns:
    1st: Y coordinate of origin (top left when we look at image)
    2nd: X coordinate of origin (top left when we look at image)
    3rd: Height of bbox
    4th: Width of bbox
    """
    return (b[1], b[0], b[3] - b[1], b[2] - b[0])


# In[26]:


def draw_rect(ax, b):
    """
    Draws bounding box rectangle on image. (also uses draw_outline() to make white bbox with black outline)
    
    Input:
    b: output of bb_hw
    
    Returns:
    4 args of Rectance = (x,y) , width, height
    """
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:],  fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


# In[27]:


def draw_text(ax, xy, txt, sz=14):
    """
    Draws text on image
    
    Input:
    ax: Output of show_img() function
    xy: x,y coordinates of image
    txt: text we want to show. generally name of class like "car"
    sz = fontsize
    """
    
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


# Let's get some image. To make our life easier, let's give strings a name. So that we can use `tab` to complete the sentences. Also let's make dictionary of 

# In[31]:


image_sample = train_json['images'][0]['file_name'] 


# In[32]:


image_sample 


# In[33]:


image_sample_id = train_json['images'][0]['id']


# In[34]:


image_sample_id


# Let's save `annotations` (bbox data) and `class` for each image in a dictionary for easy mapping

# In[28]:


# get bbox and object name for this image id
train_annotation = collections.defaultdict(lambda : []) # 

for i in train_json['annotations']:
    if not i['ignore']:
        bb = i['bbox']
        bb = np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1]) # converting to 4 corner coordinates 
        train_annotation[i['image_id']].append((bb, i['category_id']))


# In[35]:


train_annotation[image_sample_id] # here 12 is image id, 7 is category id


# In[36]:


train_annotation[image_sample_id][0]


# In[37]:


category_id = train_annotation[image_sample_id][0][1] - 1 # -1 because category ids start from 1. python index starts from 0


# In[38]:


IMAGE_PATH/'000012.jpg'


# In[39]:


im = open_image(IMAGE_PATH/image_sample)


# In[40]:


train_annotation[image_sample_id][0][0]


# Ok. let's wrap it all up. i.e function to draw an image in a function

# In[41]:


def draw_im(im, ann):
    """
    Input:
    im: Image array
    ann: Array of annotations (bbox + category_id) as saved in "train_annotations" dictionary

    """
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        b = bb_hw(b) # converting back to height width
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)


# In[42]:


# dictionary object to index image name. used to open image array

train_imagename = dict((o['id'], o['file_name']) for o in train_json['images'])

# dictionary to map id with category name
cats = dict((o['id'], o['name']) for o in train_json['categories'])


# In[43]:


def draw_idx(i, annotation_dict = train_annotation):
    """
    Input: 
    i: Image index we want to display
    annotation_dict: dictionary of annotation used (defining here as in next section we will used different one)
    
    Returns:
    Image with text and bbox
    """
    
    image_a = annotation_dict[i] # annotation of image index i
    file_name = train_imagename[i]
    im = open_image(IMAGE_PATH/file_name)
    print(im.shape)
    if annotation_dict == train_annotation: draw_im(im, image_a) # some formatting issue while plotting next one
    elif annotation_dict == train_largest_annotation: draw_im(im, [(image_a)])


# In[44]:


train_annotation[17]


# In[247]:


draw_idx(17)


# ### Task 1. Image Classification

# As talked in the blog, first we would like to classify **largest** object in the image. This is a basic **Convolution Neural Network** problem, which will have softmax layer at the end. The output will give the probability of input image being in one of the classes

# In[248]:


Image('image_class.png')


# To be clear, I am going to use `fast.ai` functions to learn network for this problem (actually transfer learning). Infact all the codes are taken from fast.ai dl part 2 course. The fast.ai library is built on top of `PyTorch`

# * **Some things to keep in mind**: 
#     * We need to classify largest object. But in train_json, we have data about all the bounding boxes. But we can find which one has largest area and train on that
#     * We are going to do transfer learning using ResNEt

# #### Getting largest object

# In[249]:


train_annotation[23]


# In[250]:


draw_idx(23)


# 2 bikes and 3 person are recognised. But let's just focus on largest object, which is this person

# In[251]:


[i for i in train_json['annotations'] if i['image_id'] == 1309]


# In[45]:


# Largest object in each image

def get_largest_bb(bb_arr):
    """
    Returns only 1 bbox which has largest area
    Area will be height*width. We have already saved height and width in 3rd and 4th elements in train_annotations. 
    Just multiply those
    """
    return bb_arr[np.argmax([bb_hw(bb[0])[2]*bb_hw(bb[0])[3] for bb in bb_arr ])]
    


# Making new array `train_largest_annotation` which has only 1 largest bbox

# {
#     IMG_ID : largest bounding box,
#     ...
# }

# In[253]:


train_annotation[1309]


# In[254]:


get_largest_bb(train_annotation[1309])


# In[46]:


train_largest_annotation = {a: get_largest_bb(b) for a,b in train_annotation.items()}


# In[47]:


train_largest_annotation[1309]


# In[48]:


draw_idx(1309, train_annotation)


# In[257]:


draw_idx(1309, train_largest_annotation)


# Saving this data about largest object in a dataframe/csv for future reference. Also because fast.ai library takes path of `csv` for input

# In[49]:


(PATH/'tmp').mkdir(exist_ok=True)
CSV = PATH/'tmp/largest.csv'


# In[50]:


train_ids = list(train_largest_annotation.keys())


# In[51]:


train_file_id = dict((i['id'], i['file_name']) for i in train_json['images'])


# In[52]:


train_cats = dict((i['id'],i['name']) for i in train_json['categories'])


# In[53]:


df = pd.DataFrame({'filename': [train_file_id[o] for o in train_ids],
                   'category_id': [train_cats[train_largest_annotation[o][1]] for o in train_ids]}, columns=['filename','category_id'])
df.to_csv(CSV, index=False)


# #### Processing image to be fed in to model (Squishing)

# We are not cropping the image as we are finally going to be interested in bounding box over all the objects in image. We might lose objects some part of image if we crop before feeding into model. But hwat we can do is **squish** the image into dimensions we need/ our model is trained on. 

# in fast.ai, `CropType.No` does that

# In[310]:


f_model = resnet34
sz = 224
bs = 64


# In[312]:


tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)
md = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms)


# `md` is pytorch model object. let's look at single batch of data form this. It has in itself seperated validatio and training set from given data

# In[318]:


x,y = next(iter(md.val_dl))


# In[320]:


x.shape # passed 64 images , all 224*224


# In[322]:


show_img(md.val_ds.denorm(to_np(x))[1]); 


# #### Re-training ResNet weights for classification 

# In[323]:


learn = ConvLearner.pretrained(f_model, md, metrics=[accuracy]) # resnet34 as pretrained model
learn.opt_fn = optim.Adam # Adam optimizers


# **What is below `lr_find` exactly doing? **  
# It is trying various values of learning rate, i.e. increasing the learning rate after each mini batch only for 1 epoch. The point on x axis (leanring_rate) after which the loss starts to increase gives us hint of best learning rate. But we won't chose learning_rate at the point of minimum, but to be conservative, little smaller than than as that point also has some potential to shoot up the loss

# In[327]:


lrf=learn.lr_find(1e-5,100)


# In[328]:


learn.sched.plot()


# ** What is plot not showing a point of minima? **  
# Probably last epoch is shooting up the learning rate much above the limits of y axis. Similar with inital few mini batches where we chose some random initialization which might be bad. So we can clip last and first 5 mini batches to look at the plot again

# In[330]:


learn.sched.plot(n_skip=5, n_skip_end=1)


# Cool. Here we have found a minima. So, better choice is to chose `learning_rate` somewhere before the point of minimum loss. Let's chose **2e-2**

# In[331]:


lr = 2e-2


# In[332]:


learn.fit(lr, 1, cycle_len=1)


# Now until now, we have updates weights for all the layers a little bit based on our new data. Let's chance our weights on final 2 layers more by freezing all layers before. 

# In[350]:


learn.freeze_to(-2)


# In[351]:


lrf=learn.lr_find(lrs/1000)
learn.sched.plot(1)


# Fitting model with new learning rate. Let's take 1e-3 and use different learning rate for different top layers. We will train the top most layer more than layer below. So, giving very small lr for top layer and little larger for layers below

# In[354]:


lrs = np.array([lr/1000,lr/100,lr])


# In[355]:


learn.fit(lrs/5, 1, cycle_len=1)


# Saving model

# In[356]:


learn.save('clas_one')


# In[357]:


learn.load('clas_one')


# In[365]:


y = learn.predict()
x,_ = next(iter(md.val_dl))


# In[368]:


#x,y = to_np(x),to_np(y)
preds = np.argmax(y, -1)


# In[369]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=md.val_ds.denorm(x)[i]
    b = md.classes[preds[i]]
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0,0), b)
plt.tight_layout()


# Ok Now we have a model that detects the largest object in an image. i.e. **Image Classification**

# ### Task 2: Single (largest) Object Detection

# In[370]:


Image('image_local.png')


# Now that we have classified the object in class, how about building a bounding box over it (**Object Localization**). The logic (as discussed in the blog also) is to add bounding box corner coordinates as label in output layer, together with class of object. Lets change the last layer of our `ResNet model`. 
# 
# But, it's already done in fastai and can be called by using **continuous=True** 

# #### Making df for bbox corners

# In[54]:


BB_CSV = PATH/'tmp/largest_bbox.csv'


# In[373]:


train_largest_annotation[12]


# In[387]:


' '.join(str(o) for o in l2)


# In[381]:


l2 = list(train_largest_annotation[12][0])


# In[389]:


df2 = pd.DataFrame({'filename': [train_file_id[o] for o in train_ids],
                   'bbox': [' '.join(str(i) for i in list(train_largest_annotation[o][0])) for o in train_ids]}, columns=['filename','bbox'])
df2.to_csv(BB_CSV, index=False)


# In[390]:


BB_CSV.open().readlines()[:3]


# #### Updating neural net for bbox prediction

# `Continuous = True` won't make a softmax layer at the end and will use MSE as criterian rather than log loss.   
# `TfmType.COORD` Transform coordinates as it flips/augments image, otherwise fixing bbox coordinates will remain absolute, we need relative positions

# In[55]:


f_model=resnet34
sz=224
bs=64

val_idxs = get_cv_idxs(len(train_file_id))


# In[56]:


tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms,
                                  
    continuous=True, num_workers=4, val_idxs=val_idxs)


# In[57]:


# aug_tfms = [RandomRotate(10, tfm_y=TfmType.COORD),
#             RandomLighting(0.05, 0.05, tfm_y=TfmType.COORD),
#             RandomFlip(tfm_y=TfmType.COORD)]

# tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=aug_tfms)

# md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True, num_workers=4)


# In[58]:


md2 = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms_from_model(f_model, sz))


# To make our model output both **class and bounding box**, we combine y of both dataloaders. (1 with class output, another with bbox output) in a tuple

# In[59]:


class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2): 
        self.ds,self.y2 = ds,y2
        self.sz = ds.sz
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x,y = self.ds[i]
        return (x, (y,self.y2[i]))


# In[60]:


trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_y)
val_ds2 = ConcatLblDataset(md.val_ds, md2.val_y)
md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2


# In[61]:


x,y = next(iter(md.val_dl))


# In[536]:


ima=md.val_ds.ds.denorm(to_np(x))[13]
b = bb_hw(to_np(y[0][13])); b  # y[0] is bbox


# In[537]:


ax = show_img(ima)
draw_rect(ax, b)
draw_text(ax, b[:2], md2.classes[y[1][13]])


# We need output like above

# We are adding a **custom_head** on the top of ResNet. This includes a few RELU, Dropouts, Linear and Batchnorm. Which will convert second final layer with 25088 neurons to final layer with 4 neurons

# In[62]:


# head_reg4 = nn.Sequential(Flatten(), nn.Linear(25088,4+len(train_cats)))
head_reg4 = nn.Sequential(
    Flatten(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(25088,256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256,4+len(cats)),
)

models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)

learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
#learn.crit = nn.L1Loss() # we now need MAE as loss (not log loss as in previous case)


# Now we will have to define a **custom loss** for this data loader as y is now a tuple 

# In[565]:


y[0].shape, y[1].shape


# In[63]:


def detn_loss(input, target):
    bb_t,c_t = target # bbox, class
    bb_i,c_i = input[:, :4], input[:, 4:]
    bb_i = F.sigmoid(bb_i)*sz
    return F.l1_loss(bb_i, bb_t) + F.cross_entropy(c_i, c_t)*20 # 20 is just to scale cross entropy with l1loss values

def detn_l1(input, target):
    bb_t,_ = target
    bb_i = input[:, :4]
    bb_i = F.sigmoid(bb_i)*224
    return F.l1_loss(V(bb_i),V(bb_t)).data

def detn_acc(input, target):
    _,c_t = target
    c_i = input[:, 4:]
    return accuracy(c_i, c_t)

learn.crit = detn_loss
learn.metrics = [detn_acc, detn_l1]


# In[64]:


# updating criteria and metrics in learn now

learn.crit = detn_loss
learn.metrics = [detn_acc, detn_l1]


# Let's check our convnet

# In[568]:


learn.summary()


# Final linear layer has been added which gives 4 outputs (bbox corners)

# Finding learning rate again as we did in previous case

# #### Fitting model

# In[569]:


learn.lr_find() # start, end
learn.sched.plot() 


# We can chose `2*e-3` as our learnign rate with logic as discussed before

# In[570]:


lr = 1e-2


# In[571]:


learn.fit(lr, 2, cycle_len=1, cycle_mult=2) # total 3 epochs. (2 cycles) Don't worry about cycle_len, cycle_mul for now


# This can be trained more by **freezing** all layers except last 2-3 and decreasing the learning date. I am going to paste the code to do that. But I am running this nb on CPU. So I would skip running it. 

# In[1]:


# learn.freeze_to(-3)


# In[573]:


# lrs = np.array([lr/100,lr/10,lr])


# In[574]:


# learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)


# Saving model

# In[575]:


learn.save('box_one')


# In[65]:


learn.load('box_one')


# Visualizing results

# In[66]:


y = learn.predict()
x,_ = next(iter(md.val_dl))


# In[ ]:


from scipy.special import expit


# In[68]:


fig, axes = plt.subplots(3, 4, figsize=(15, 10))
for i,ax in enumerate(axes.flat):
    ima=md.val_ds.ds.denorm(to_np(x))[i]
    bb = expit(y[i][:4])*224
    b = bb_hw(bb)
    c = np.argmax(y[i][4:])
    ax = show_img(ima, ax=ax)
    draw_rect(ax, b)
    draw_text(ax, b[:2], md2.classes[c])


# ### Task 3 : Multiple object localization (Next notebook)
