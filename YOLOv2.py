import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os
import dataload
import joblib
import random
import pylab


def getbatch(dataloader, batchsize, flag):
    img_batch = torch.zeros((batchsize, 3, inputsize, inputsize))
    bndbox_batch = []
    boxlabel_batch = []
    i = 0
    j = 0
    while True:
        if i-j == batchsize:
            break 
        img, bndbox, boxlabel = dataload.Dataloader(dataloader, flag*batchsize+i)
        i += 1
        bndbox = torch.LongTensor(bndbox)
        img = img.float()
        if (bndbox[:,2]-bndbox[:,0]<=0).sum()>0 or (bndbox[:,3]-bndbox[:,1]<=0).sum()>0:
            j += 1
#             print('Abnormal data!')
            continue
        _, H, W = img.shape
        img  = cv2.resize(img.permute(1,2,0).numpy(), (inputsize, inputsize))
        bndbox = dataload.resize_box(bndbox.tolist(), (H, W), (inputsize, inputsize))
        bndbox = torch.from_numpy(bndbox).float()
        bndbox[bndbox<0] = 0
        bndbox[bndbox>inputsize] = inputsize
        img_batch[i-j-1] = torch.from_numpy(img).permute(2,0,1)
        bndbox_batch.append(bndbox)
        boxlabel_batch.append(boxlabel)
        
    return img_batch, bndbox_batch, boxlabel_batch


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class DarkNet_19(nn.Module):
    def __init__(self, num_classes=3):        
        super(DarkNet_19, self).__init__()
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2,2), 2),
        )
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2,2), 2)
        )
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2,2), 2)
        )
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        C_4 = self.conv_4(x)
        C_5 = self.conv_5(self.maxpool_4(C_4))
        C_6 = self.conv_6(self.maxpool_5(C_5))
        return C_5, C_6
    
def darknet19(**kwargs):
    model = DarkNet_19()
    path_to_dir = os.path.dirname(os.path.abspath('__file__'))
    model.load_state_dict(torch.load(path_to_dir + '/darknet19_hr_75.52_92.73.pth', map_location='cuda'), strict=False)
    return model


class my_YOLOv2(nn.Module):
    def __init__(self):
        super(my_YOLOv2, self).__init__()
        self.pretrained = darknet19()
        self.fine_grained = Conv_BN_LeakyReLU(512, 64, 1)
        self.conv1 = Conv_BN_LeakyReLU(1024, 1024, 3, 1)
        self.conv2 = Conv_BN_LeakyReLU(1024+256, 1024, 3, 1)  
        self.conv3 = nn.Conv2d(1024, k*(5+C), 1, 1, 0)
    def forward(self, x):
        fg, output = self.pretrained(x)
        fg = self.fine_grained(fg)
        batchsize, c, h, w = fg.data.size()
        
        fg = fg.view(batchsize, int(c/4), h, 2, w, 2).permute(0, 3, 5, 1, 2, 4).contiguous()
        fg = fg.view(batchsize, -1, int(h/2), int(w/2))
        output = self.conv1(output)
        output = torch.cat((output, fg), 1)
        output = self.conv2(output)
        output = self.conv3(output)
        return output

    
def makeroi(bndbox):
    box = xy2wh(bndbox).squeeze()/sampling
    ctrx = box[0]
    ctry = box[1]
    w = box[2]
    h = box[3]
    newbox = torch.Tensor([0,0,w,h])
    indx = ctrx.long()
    indy = ctry.long()
    anchor_ = torch.zeros((len(anchorf), 4))
    for i in range(len(anchor)):
        anchor_[i,:] = torch.cat((torch.Tensor([0,0]), anchorf[i]), 0)
    iou = my_IoU(anchor_, newbox)
    indmax = iou.argmax()
    if (iou <= thr_pos).all():
        pwh = anchorf[indmax]
        tx = ctrx - ctrx.long()
        ty = ctry - ctry.long()
        tw = torch.log(w/pwh[0])
        th = torch.log(h/pwh[1])
        weight = 2-w*h*((sampling/inputsize)**2)
        delta = torch.FloatTensor([tx,ty,tw,th])
        roi = torch.cat(([torch.FloatTensor([indmax, indx, indy]), delta, weight.unsqueeze(0), bndbox.float()/inputsize]),0).tolist()
        return roi
    else:
        roi = []
        for i in range(len(iou)):
            if iou[i]>thr_pos:
                if i == indmax:
                    pwh = anchorf[indmax]
                    tx = ctrx - ctrx.long()
                    ty = ctry - ctry.long()
                    tw = torch.log(w/pwh[0])
                    th = torch.log(h/pwh[1])
                    weight = 2-w*h*((sampling/inputsize)**2)
                    delta = torch.FloatTensor([tx,ty,tw,th])
                    roi.append(torch.cat(([torch.FloatTensor([indmax, indx, indy]), delta, weight.unsqueeze(0), bndbox.float()/inputsize]),0).tolist())
                else:
                    roi.append(torch.cat(([torch.FloatTensor([indmax, indx, indy]), torch.zeros(4).float(), torch.FloatTensor([-1]), torch.zeros(4).float()]),0).tolist())
        return roi 
    
def maketarget(bndbox, boxlabel):
    target = torch.zeros((batchsize, Hf, Wf, k, 1+1+4+1+4)) #[batch,13,13,5,11]
    # objectness // cls // tx, ty, tw, th // weight(sig(t0)) // x1,y1,x2,y2 
    for i in range(batchsize):
        for j in range(len(bndbox[i])):
            box = bndbox[i][j]
            clslabel = boxlabel[i][j]
            roi = makeroi(box)
            if torch.Tensor(roi).ndim == 1: roi = [roi]
            for r in roi:
                indmax, indx, indy, tx, ty, tw, th, weight, x1, y1, x2, y2 = r
                indmax, indx, indy = int(indmax), int(indx), int(indy)
                if weight > 0.:
                    target[i, indy, indx, indmax, 0] = 1
                    target[i, indy, indx, indmax, 1] = clslabel
                    target[i, indy, indx, indmax, 2:6] = torch.Tensor([tx,ty,tw,th])
                    target[i, indy, indx, indmax, 6] = weight
                    target[i, indy, indx, indmax, 7:] = torch.Tensor([x1,y1,x2,y2])
                else:
                    target[i, indy, indx, indmax, 0] = -1
                    target[i, indy, indx, indmax, 6] = -1

    target = target.view(batchsize, Hf*Wf*k, -1)
    return target


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    iou = area_i / (area_a + area_b - area_i)
    iou[torch.isnan(iou)] = 0
    return iou

class MSELoss(nn.Module):
    def __init__(self,  weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, mask):
        # We ignore those whose tarhets == -1.0. 
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss

def loss(pred_conf, pred_cls, pred_txtytwth, label, num_classes):
    conf_loss_function = MSELoss(reduction='mean') 

    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    pred_cls = pred_cls.permute(0, 2, 1)
    txty_pred = pred_txtytwth[:, :, :2]
    twth_pred = pred_txtytwth[:, :, 2:]
        
    gt_conf = label[:, :, 0].float()
    gt_obj = label[:, :, 1].float()
    gt_cls = label[:, :, 2].long()
    gt_txtytwth = label[:, :, 3:-1].float()
    gt_box_scale_weight = label[:, :, -1]
    gt_mask = (gt_box_scale_weight > 0).float()

    # objectness loss
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)
    conf_loss = lambda_obj * pos_loss + lambda_noobj * neg_loss
    
    # class loss
    cls_loss = torch.mean(torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask, 1))
    
    # box loss
    txty_loss = torch.mean(torch.sum(torch.sum(twth_loss_function(txty_pred, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight * gt_mask, 1))
    twth_loss = torch.mean(torch.sum(torch.sum(twth_loss_function(twth_pred, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight * gt_mask, 1))

    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + lambda_cls*cls_loss + lambda_coor*txtytwth_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss


def my_IoU(anchor, gt):
    if anchor.ndim == 1:
        anchor = anchor.unsqueeze(0)
    if gt.ndim == 1:
        gt = gt.unsqueeze(0)
    anchor = np.array(anchor.tolist())
    gt = np.array(gt.tolist())
    IoU = np.zeros((len(anchor),len(gt)))
    for i in range(len(gt)):                 
        IoU_W = np.maximum(np.min((anchor[:,0], anchor[:,2], gt[i,0]*np.ones(len(anchor)), gt[i,2]*np.ones(len(anchor))),0) + anchor[:,2]-anchor[:,0] + gt[i,2]-gt[i,0] - np.max((anchor[:,0], anchor[:,2], gt[i,0]*np.ones(len(anchor)), gt[i,2]*np.ones(len(anchor))), 0), 1e-100)
        IoU_H = np.maximum(np.min((anchor[:,1], anchor[:,3], gt[i,1]*np.ones(len(anchor)), gt[i,3]*np.ones(len(anchor))),0) + anchor[:,3]-anchor[:,1] + gt[i,3]-gt[i,1] - np.max((anchor[:,1], anchor[:,3], gt[i,1]*np.ones(len(anchor)), gt[i,3]*np.ones(len(anchor))), 0), 1e-100)
        IoU[:,i] = (IoU_W*IoU_H)/((anchor[:,3]-anchor[:,1])*(anchor[:,2]-anchor[:,0]) + (gt[i,3]-gt[i,1])*(gt[i,2]-gt[i,0]) - IoU_W*IoU_H)
        for j in range(len(anchor)):
                if (gt[i] == anchor[j]).all():
                    IoU[j,i] = 1
    IoU = torch.Tensor(IoU)
    IoU[(IoU>=0)&(IoU<=1)==False] = 0
    return IoU.squeeze()


def xy2wh(anchor):
    if anchor.ndim == 1:
        anchor = anchor.unsqueeze(0)
    w = anchor[:,2]-anchor[:,0]
    h = anchor[:,3]-anchor[:,1]     
    w[w<=0] = 1e-100
    h[h<=0] = 1e-100
    
    ctrx = anchor[:,0]+w/2
    ctry = anchor[:,1]+h/2
    xywh = torch.stack((ctrx,ctry,w,h),1)
    
    return xywh


def wh2xy(anchor):
    if anchor.ndim == 1:
        anchor = anchor.unsqueeze(0)
    if anchor.ndim == 2:
        anchor[:,2][anchor[:,2]<=0] = 1e-100
        anchor[:,3][anchor[:,3]<=0] = 1e-100
        xy = torch.zeros(anchor.shape)
        xy[:,0] = anchor[:,0]-anchor[:,2]/2 #x1
        xy[:,2] = anchor[:,0]+anchor[:,2]/2 #x2
        xy[:,1] = anchor[:,1]-anchor[:,3]/2 #y1
        xy[:,3] = anchor[:,1]+anchor[:,3]/2 #y2
    else:
        anchor[:,:,2][anchor[:,:,2]<=0] = 1e-100
        anchor[:,:,3][anchor[:,:,3]<=0] = 1e-100
        xy = torch.zeros(anchor.shape)
        xy[:,:,0] = anchor[:,:,0]-anchor[:,:,2]/2 #x1
        xy[:,:,2] = anchor[:,:,0]+anchor[:,:,2]/2 #x2
        xy[:,:,1] = anchor[:,:,1]-anchor[:,:,3]/2 #y1
        xy[:,:,3] = anchor[:,:,1]+anchor[:,:,3]/2 #y2
    return xy

anchor = [[0.5650, 0.6158],
        [0.0787, 0.1464],
        [0.1828, 0.2954],
        [0.2975, 0.4013],
        [0.0000, 0.2476]]

anchor = torch.Tensor(anchor)
anchorf = anchor*13

k = 5
C = 3
sampling = 32
thr_pos = 0.6
inputsize = 416

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flag = 0
batchsize = 5
epochs = 200
lr = 5e-4
train_flag = 1600
valid_flag = 1760
test_flag = 2030
dataloader = dataload.load_data('./data')
dirs = './'
def loadmodel(pretrained=True):
    if pretrained == True:
        yolo = joblib.load(dirs+'/model.pkl')
        yolo = yolo.to(device)
    else:
        yolo = my_YOLOv2().to(device)
    optimizer = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    return yolo, optimizer, scheduler


lambda_obj = 5
lambda_noobj = 1
lambda_cls = 0.5
lambda_coor = 1

###########################################
############### Training ##################
###########################################

Hf, Wf = [13,13]
def train():
    yolo, optimizer, scheduler = loadmodel(pretrained=False)
    loss_ = []
    for epoch in range(epochs):
        st = time.time()
        print('='*100)
        print('epoch:', epoch)
        for flag in range(train_flag//batchsize):
    #         if flag % 10 == 0: inputsize = random.choice(inputsizelist)
            optimizer.zero_grad()
            img, box, label = getbatch(dataloader, batchsize, flag)
            img = img.to(device)
            output = yolo(img)

            B, Cn, Hf, Wf = output.size()
            pred = output.permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, Cn) 

            pred_conf = pred[:,:,:k].contiguous().view(B, Hf*Wf*k, -1).to(device) # [batch,845,1] 
            pred_cls = pred[:,:,k:(1+C)*k].contiguous().view(B, Hf*Wf*k, -1).to(device) # [batch,845,C] 
            pred_delta = pred[:,:,(1+C)*k:].contiguous().view(B, Hf*Wf, k, -1).to(device) # [batch,169,5,4] 

            y, x = torch.meshgrid([torch.arange(Hf),torch.arange(Wf)])
            xy = torch.stack([x,y], 2).float().to(device) 
            pwh = anchorf.repeat(Hf*Wf, 1, 1).unsqueeze(0).to(device) # [1,169,5,2] 
            bxy = torch.sigmoid(pred_delta[:,:,:,:2])+xy.view(1, Hf*Wf, 1, 2) 
            bwh = pwh*torch.exp(pred_delta[:,:,:,2:4]) 
            b = torch.cat((bxy, bwh), 3).view(B, Hf*Wf*k, -1) # [batch,845,4] 
            with torch.no_grad(): 
                pred_coor = wh2xy(b*sampling/inputsize).view(-1,4) # [batch*845,4] 0~1 normalization
            pred_delta = pred_delta.view(B, Hf*Wf*k, -1) # txtytwth

            target = maketarget(box, label) # objectness // cls // tx, ty, tw, th // weight(sig(t0)) // x1,y1,x2,y2  
            gt_ = target[:,:,7:].view(-1, 4) # gt x1y1x2y2
            iou = iou_score(pred_coor, gt_) 
            target = torch.cat((iou.view(batchsize, Hf*Wf*k, 1), target[:,:,:7]), -1)
            conf_loss, cls_loss, txtytwth_loss, total_loss = loss(pred_conf=pred_conf.to(device), pred_cls=pred_cls.to(device),
                                                                                    pred_txtytwth=pred_delta.to(device),
                                                                                    label=target.to(device),
                                                                                    num_classes=C)
            loss_.append(total_loss.item())
            print('batch: %d / %d '%(flag, train_flag//batchsize))
            print('loss: %.4f (conf: %.4f, cls: %.4f, coor: %.4f)'% (total_loss.item(), conf_loss.item(), cls_loss.item(), txtytwth_loss.item()))
            total_loss.backward()
            nn.utils.clip_grad_norm_(yolo.parameters(), 1e-2)
            optimizer.step()
        scheduler.step
        ed = time.time()
        print('time: %.2f s' %(ed-st))

###########################################
############# Prediction ##################
###########################################


dataloader = dataload.load_data('./data', train=False)
def prediction(randpic=False):
    yolo, optimizer, scheduler = loadmodel(pretrained=True)
    picnum = 0
#     for pic in range(picnum, picnum+3):
    while True:
        if picnum == 3: break
        if randpic == True:
            pic = random.randint(valid_flag, test_flag)  
        else:
            if picnum == 0:
                pic = 1850
            else:
                pic += 1
        img, box, label = getbatch(dataloader, 1, pic)
        img = img.to(device)
        st = time.time()
        output = yolo(img)

        B, Cn, Hf, Wf = output.size()
        pred = output.permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, Cn) 

        pred_conf = pred[:,:,:k].contiguous().view(B, Hf*Wf*k, -1).to(device) 
        pred_cls = pred[:,:,k:(1+C)*k].contiguous().view(B, Hf*Wf*k, -1).to(device) 
        pred_delta = pred[:,:,(1+C)*k:].contiguous().view(B, Hf*Wf, k, -1).to(device) 
        y, x = torch.meshgrid([torch.arange(Hf),torch.arange(Wf)])
        xy = torch.stack([x,y], 2).float().to(device) 
        pwh = anchorf.repeat(Hf*Wf, 1, 1).unsqueeze(0).to(device)  
        bxy = torch.sigmoid(pred_delta[:,:,:,:2])+xy.view(1, Hf*Wf, 1, 2)
        bwh = pwh*torch.exp(pred_delta[:,:,:,2:4]) 
        b = torch.cat((bxy, bwh), 3).view(B, Hf*Wf*k, -1) 
        predbox = wh2xy(b*sampling/inputsize).view(-1,4) 
        predconf = torch.sigmoid(pred_conf)[0]
        predscore = torch.softmax(pred_cls[0,:,:], 1)*predconf

        lbl = predscore.argmax(1)
        indsort = predscore[torch.arange(len(predscore)),lbl].argsort(descending=True)
        thrscore = torch.where(predconf.squeeze().sort(descending=True)[0]>0.05)[0]
        if torch.Size(thrscore) == torch.Size([]):
            predlabel = lbl[indsort][:1]
            box_nms = predbox[indsort][:1]
            predconf = predconf[indsort][:1]
        else:
            predlabel = lbl[indsort][thrscore]
            box_nms = predbox[indsort][thrscore]
            predconf = predconf[indsort][thrscore]

        predbox = box_nms*inputsize

        maxind = torch.arange(len(box_nms))
        ind_nms = []
        thr_nms = 0.1
        box_nms_ = box_nms.clone()
        while(len(box_nms_)>=1):
            gt_nms = box_nms_[0]
            ind_nms.append(maxind[0])
            iou_nms = my_IoU(box_nms_, gt_nms)
            box_nms_ = box_nms_[(iou_nms<thr_nms).squeeze()]
            maxind = maxind[(iou_nms<thr_nms).squeeze()]
        predbox = box_nms[torch.LongTensor(ind_nms)][:5]*inputsize
        predlabel = predlabel[torch.LongTensor(ind_nms)][:5]
        predconf = predconf[torch.LongTensor(ind_nms)][:5]

        ed = time.time()
#         print('time: %.2f s' %(ed-st))

#         print(predbox, predlabel)

        ##
        img, bndbox, boxlabel = dataload.Dataloader(dataloader, pic)
        if (bndbox[:,2]-bndbox[:,0]<=0).sum()>0 or (bndbox[:,3]-bndbox[:,1]<=0).sum()>0:
            print('Abnormal data!')
            continue
        else:
            picnum += 1
            boxlabel = torch.ones(len(bndbox))
            bndbox = torch.LongTensor(bndbox)
            img = img.float()

        _, H, W = img.shape
        img  = cv2.resize(img.permute(1,2,0).numpy(), (inputsize,inputsize))
        bndbox = dataload.resize_box(bndbox.tolist(), (H, W), (inputsize, inputsize))
        bndbox = torch.from_numpy(bndbox).float()

#         for j in range(len(bndbox)):
#             cv2.rectangle(img, (bndbox[j][0], bndbox[j][1]), (bndbox[j][2], bndbox[j][3]), (0,255,255), 2)
#             text = 'face'
        #     cv2.putText(img, text, (bndbox[j][0],bndbox[j][1]), 1, 1.5, (0,255,255), 2)
        for j in range(len(predbox)):
            cv2.rectangle(img, (predbox[j][0], predbox[j][1]), (predbox[j][2], predbox[j][3]), (0,255,0), 2)
            score = '%.2f%%' %(predconf[j]*100)
            cv2.putText(img, score, (predbox[j][0],predbox[j][1]), 0, 1, (0,255,0), 2)
        img = dataload.Unnormalize_Orgsizeimg(img, (H, W))
        plt.figure(figsize = (10,10))
        plt.imshow(img)
        pylab.show()
# prediction(randpic=True)
# prediction()

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, threshold):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    # set_trace()
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) 
        #== len(true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0).to(device)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0).to(device).squeeze()  # (n_objects)
    # true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0).to(device)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0).to(device)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0).to(device)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    precisions_classes = []
    for c in range(1, n_classes):

        # Extract only objects with this class
#         print(true_images, true_images.shape, true_labels, true_labels.shape)
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        # true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        #n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects
        n_objects = true_class_images.size(0)
        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        # set_trace()
        #true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
        #    device)  # (n_class_objects)
        true_class_boxes_detected = torch.zeros((true_class_images.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)


        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            print("no detection of class %d is found" %c)
            precisions_classes.append(torch.zeros((11), dtype=torch.float).to(device))
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            #object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = my_IoU(this_detection_box.squeeze(), object_boxes)  # (1, n_class_objects_in_img)

            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > threshold:
                # If the object it matched with is 'difficult', ignore it
                #if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    #if true_class_boxes_detected[original_ind] == 0:
                     #   true_positives[d] = 1
                     #   true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    #else:
                     #   false_positives[d] = 1
                if true_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                # Otherwise, it's a false positive (since this object is already accounted for)
                else:
                    false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        #cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        cumul_recall = cumul_true_positives / n_objects


        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]
        precisions_classes.append(precisions)


    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return precisions_classes, average_precisions, mean_average_precision

# yolo, optimizer, scheduler = loadmodel(pretrained=True)
# PR = []
# thr = 0.1
# det_boxes = []
# det_labels = []
# det_scores = []
# true_boxes = []
# true_labels = []
# for pic in range(valid_flag, test_flag):
#     img, box, label = getbatch(dataloader,1, pic)
#     img = img.to(device)
#     output = yolo(img)

#     true_boxes.append(box[0])
#     true_labels.append(label[0])

#     B, Cn, Hf, Wf = output.size()
#     pred = output.permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, Cn) 

#     pred_conf = pred[:,:,:k].contiguous().view(B, Hf*Wf*k, -1).to(device) 
#     pred_cls = pred[:,:,k:(1+C)*k].contiguous().view(B, Hf*Wf*k, -1).to(device)
#     pred_delta = pred[:,:,(1+C)*k:].contiguous().view(B, Hf*Wf, k, -1).to(device) 
#     y, x = torch.meshgrid([torch.arange(Hf),torch.arange(Wf)])
#     xy = torch.stack([x,y], 2).float().to(device) 
#     pwh = anchorf.repeat(Hf*Wf, 1, 1).unsqueeze(0).to(device) 
#     bxy = torch.sigmoid(pred_delta[:,:,:,:2])+xy.view(1, Hf*Wf, 1, 2) 
#     bwh = pwh*torch.exp(pred_delta[:,:,:,2:4])
#     b = torch.cat((bxy, bwh), 3).view(B, Hf*Wf*k, -1)
#     predbox = wh2xy(b*sampling/inputsize).view(-1,4) 
#     predconf = torch.sigmoid(pred_conf)[0]
#     predscore = torch.softmax(pred_cls[0,:,:], 1)*predconf

#     lbl = predscore.argmax(1)
#     indsort = predscore[torch.arange(len(predscore)),lbl].argsort(descending=True)
#     thrscore = torch.where(predconf.squeeze().sort(descending=True)[0]>0.05)[0]
#     if torch.Size(thrscore) == torch.Size([]):
#         predlabel = lbl[indsort][:1]
#         box_nms = predbox[indsort][:1]
#         predconf = predconf[indsort][:1]
#     else:
#         predlabel = lbl[indsort][thrscore]
#         box_nms = predbox[indsort][thrscore]
#         predconf = predconf[indsort][thrscore]

#     predbox = box_nms*inputsize

#     maxind = torch.arange(len(box_nms))
#     ind_nms = []
#     thr_nms = 0.1
#     box_nms_ = box_nms.clone()
#     while(len(box_nms_)>=1):
#         gt_nms = box_nms_[0]
#         ind_nms.append(maxind[0])
#         iou_nms = my_IoU(box_nms_, gt_nms)
#         box_nms_ = box_nms_[(iou_nms<thr_nms).squeeze()]
#         maxind = maxind[(iou_nms<thr_nms).squeeze()]
#     predbox = box_nms[torch.LongTensor(ind_nms)][:5]*inputsize
#     predlabel = predlabel[torch.LongTensor(ind_nms)][:5]
#     predconf = predconf[torch.LongTensor(ind_nms)][:5]

#     det_boxes.append(predbox.detach())
#     det_labels.append(predlabel.detach())
#     det_scores.append(predconf.detach())

# # Label map
# labels = (['face'])
# label_map = {k: v + 1 for v, k in enumerate(labels)}
# label_map['background'] = 0
# rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# mAPs = {}
# APs = {}
# precisions_dict = {}
# for threshold in np.arange(0.5, 0.95, 0.05):  
#     precisions, AP, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, threshold)
# #     print(APs)
#     threshold = "%.2f" %threshold

#     mAPs[threshold] = mAP 
#     APs[threshold] = AP
#     precisions_dict[threshold] = precisions


# print(APs["0.50"])
# print("Mean Average Precision (mAP@.5): %.3f" % mAPs["0.50"])
# print(APs["0.70"])
# print("Mean Average Precision (mAP@.7): %.3f" % mAPs["0.70"])
# print(APs["0.90"])
# print("Mean Average Precision (mAP@.9): %.3f" % mAPs["0.90"])
# mean_mAPs = sum(mAPs.values())/len(mAPs)
# print("\nMean Average Precision (mAP@[.5:.95]): %.3f" % mean_mAPs)

# fig = plt.figure(figsize=(10,3))
# i = 1

# for threshold in ["0.50", "0.70", "0.90"]:
#     x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     sub = "13"+str(i)
#     plt.subplot(sub)
#     precisions_dict[threshold][0] = precisions_dict[threshold][0].cpu().numpy()
#     label_ = "threshold_" + threshold
#     plt.step(x, precisions_dict[threshold][0], label=label_)
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.legend(loc='upper right')

#     print("plotted figure threshold "+threshold)
#     i = i + 1

# figure_name = "P_R_curve_face.png"
# fig.tight_layout()
# #     print("Saving to ", figure_name)
# # fig.savefig(figure_name)
# #     print("Saved!")
# plt.show()
# plt.close()


# In[ ]:

# if __name__ == "__main__":


