# -*- coding: utf-8 -*-
"""Q4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1izHc1ijcKWCnmK8ldeuxN8JHaSxopWjt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

import pandas as pd
import numpy as np
import pickle
from cv2 import imshow as cv2_imshow


tSNE_PLOT=False
VISUALIZE=False
CLASS_DISTRI=False

# To recreate preprocessed dataset, set this to True
# To use saved preprocessed dataset, set this to False
DO_PREPROCESSING = False

LOSS_PLOT=True
ROC_PLOT=True
CONF_ACC=True



# RANDOM SEEDS
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# IMPORT GIVEN DATASET
train = pickle.load(open("Datasets/Q4/train_CIFAR.pickle", "rb"))
test = pickle.load(open("Datasets/Q4/test_CIFAR.pickle", "rb"))
Xtrain = train['X']
ytrain = train['Y']
Xtest = test['X']
ytest = test['Y']


# t-SNE PLOT
if tSNE_PLOT:
  from sklearn.manifold import TSNE
  from sklearn.decomposition import PCA

  pca = PCA(0.95).fit_transform(Xtrain)
  tsne = TSNE(n_components=2).fit_transform(pca)

  plt.figure(figsize=(20,10))

  for c in range(2):
      select = tsne[ytrain==c]
      plt.scatter(select[:,0],select[:,1],label="Class "+str(c))

  plt.suptitle("t-SNE Analysis")
  plt.legend()
  plt.xlabel("Axis-1")
  plt.ylabel("Axis-2")
  plt.savefig("Plots/Q4/tsne.png")


# CLASS DISTRIBUTION (from ASSIGN_1-2)
if CLASS_DISTRI:
  def class_distribution(y):
      """
      Input Parameter:
      
      y : class labels vector having shape as (n_samples,)

      Output: Prints the class distribution

      Returns: None
      """
      unique, counts = np.unique(y, return_counts=True)
      distri = dict(zip(unique, counts))
      print("Count\tFraction")
      total = sum(counts)
      for c in range(len(unique)):
          print(distri[c],"\t",distri[c]/total)
      print("Total:",total)

  print("Train Data Class Distribution")
  class_distribution(ytrain)
  print()
  print("Test Data Class Distribution")
  class_distribution(ytest)


# UNFLATTEN THE DATASETS
Xtrain = Xtrain.reshape(-1,3,32,32).transpose(0,2,3,1)
Xtrain.shape
Xtest = Xtest.reshape(-1,3,32,32).transpose(0,2,3,1)
Xtest.shape


# VISUALIZING A RANDOM SAMPLE FROM THE DATASET
if VISUALIZE:
  print("10 Random Train Images")
  for _ in range(10):
    i = random.randint(0,Xtrain.shape[0]-1)
    cv2_imshow(Xtrain[i])
    print(ytrain[i])

  print("10 Random Test Images")
  for _ in range(10):
    i = random.randint(0,Xtest.shape[0]-1)
    cv2_imshow(Xtest[i])
    print(ytest[i])


# INIT THE GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using",device)



if DO_PREPROCESSING:
  # FETCH PRETRAINED ALEXNET
  import torchvision.models as models
  alexnet = models.alexnet(pretrained=True)
  alexnet = alexnet.to(device)

  # PREPROCESS TRANSFORM
  pp_transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])

  # PREPROCESSING BEGINS (TRAIN DATA)
  from PIL import Image
  a=[]
  for i in range(Xtrain.shape[0]):
    img = Image.fromarray(Xtrain[i])
    img = pp_transform(img).unsqueeze(0)
    img = img.to(device)
    img = alexnet(img)[0]
    img = img.to('cpu').detach().numpy()
    a.append(img)

  df = pd.DataFrame(a)
  df[1000] = ytrain
  df.to_csv("Datasets/Q4/preprocessed_train.csv", index=False)

  # PREPROCESSING BEGINS (TEST DATA)
  from PIL import Image
  a=[]
  for i in range(Xtest.shape[0]):
    img=Image.fromarray(Xtest[i])
    img = pp_transform(img).unsqueeze(0)
    img = img.to(device)
    img = alexnet(img)[0]
    img = img.to('cpu').detach().numpy()
    a.append(img)

  df = pd.DataFrame(a)
  df[1000] = ytest
  df.to_csv("Datasets/Q4/preprocessed_test.csv", index=False)



"""USING EXTRACTED FEATURES AS INPUT TO NEURAL NET JUST LIKE Q3"""


# CUSTOM NN CLASS
class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_features, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, n_classes)
        
    def forward(self, x):
      y = self.linear1(x)
      y = self.relu1(y)
      y = self.linear2(y)
      y = self.relu2(y)
      y = self.linear3(y)
      # omit softmax at the end
      return y

# IMPORT PREPROCESSED DATASET
train = pd.read_csv("Datasets/Q4/preprocessed_train.csv")
test = pd.read_csv("Datasets/Q4/preprocessed_test.csv")

# MAKING IT COMPATIBLE WITH THE CUSTOM NN
Xtrain = torch.from_numpy(train.iloc[:,:-1].to_numpy(dtype=np.float32)).to(device)
ytrain = torch.from_numpy(train.iloc[:,-1].to_numpy()).to(device)

Xtest = torch.from_numpy(test.iloc[:,:-1].to_numpy(dtype=np.float32)).to(device)
ytest = torch.from_numpy(test.iloc[:,-1].to_numpy()).to(device)

# DECLARING MODEL and LOSS
model = MLP(1000, 2) #i=0
criterion = nn.CrossEntropyLoss()

# COPY TO GPU (IF AVAILABLE)
model = model.to(device)
criterion = criterion.to(device)

# DECLARING OPTIMIZER and EPOCHS
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100


# TRAIN LOOP
print("Now Training...")
train_loss=[]
test_loss=[]
for epoch in range(epochs):
  ypred = model(Xtrain)
  loss = criterion(ypred, ytrain)
  with torch.no_grad():
    train_loss.append(loss.detach())
    test_loss.append(criterion(model(Xtest), ytest))
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()


# LOSS PLOT
if LOSS_PLOT:
  import matplotlib.pyplot as plt
  iters = list(range(epochs))
  plt.figure(figsize=(20,10))
  plt.plot(iters, train_loss, label='train loss')
  plt.plot(iters, test_loss, '--', label='test loss')
  plt.legend()
  plt.xlabel('Number of Iterations')
  plt.ylabel('Cross Entropy Loss')
  plt.suptitle("Cross Entropy Loss Vs Epochs")
  plt.savefig("Plots/Q4/loss_vs_epoch.png")


# CONFUSION MATRIX AND TEST ACCURACY
if CONF_ACC:
  from sklearn.metrics import confusion_matrix
  import seaborn as sns

  with torch.no_grad():
    ypred = model(Xtest)
    ypred_label = ypred.argmax(1)
    test_acc = torch.mean((ypred_label == ytest).float()).item()

    cf = confusion_matrix(ytest.cpu(), ypred_label.cpu())
    sns.heatmap(cf, annot=True)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.suptitle("Q4. Test Accuracy: "+str(test_acc))
    plt.savefig("Plots/Q4/conf_matrix.png")


# ROC CURVE PLOT
if ROC_PLOT:
  from sklearn.metrics import roc_curve, auc

  with torch.no_grad():
    ypred = model(Xtest)
    ypred = nn.functional.softmax(ypred)
    
    ypred = ypred.cpu().numpy()[:,1]
    ytest = ytest.cpu().numpy()

    fpr, tpr, thresholds = roc_curve(ytest, ypred, pos_label=1)

    plt.figure(figsize=(20,10))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Base Line')
    plt.plot(fpr, tpr, linewidth=2, label='ROC Curve (AUC = '+str(auc(fpr, tpr))+')')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Q4. ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("Plots/Q4/ROC_AUC.png")
