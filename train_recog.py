from __future__ import print_function
import argparse
import os
import sys
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

import model as net
from misc import *