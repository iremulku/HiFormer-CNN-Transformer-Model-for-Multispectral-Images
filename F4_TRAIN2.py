from __future__ import print_function
import os
import torch 
import torch.nn as nn
import numpy as np
from F5_JACCARD import Jaccard
from F1_UNET_V1_1 import UNetV1
#from F9_UNET_V2_4 import UNetV2
from F9_UNET_V2_3 import UNetV2
from F10_SEGNET_V1 import SegNet
from F15_DFANET import DFANet 
from F12_DLINKNET_V3 import DinkNet101
from F20_DILATEDUNET import CamDUNet
from F21_GENERAL_UNET import R2U_Net, AttU_Net, R2AttU_Net
from F22_NESTEDUNET import NestedUNet
from F23_DULANORM_UNET import DualNorm_Unet
from F24_INCEPTION_UNET import InceptionUNet
from F25_SCAG_UNET import AttU_Net_with_scAG
from F26_FSFNet import FSFNet
from F27_LMFFNet import LMFFNet
from LMFFNet_IREM import LMFFNet2
from LMFFNet_IREM3 import LMFFNet3
from F28_FASSDNet import FASSDNet
from F29_ENet import ENet
from F30_ELANet import ELANet
#from F14_DEEPLABV3PLUS_V1 import DeepLabv3_plus
#from F14_DEEPLABV3PLUS_V1_DROP2_Resnet34 import DeepLabv3_plus
from F14_DEEPLABV3PLUS_V4_xception import DeepLabv3_plus


class Config(object):
    NAME= "dfaNet"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#dev = torch.device("cpu")
device = torch.device(dev) 


def train_model(n_epochs, trainloss, validationloss, accuracy, model, scheduler, lrFile, training_generator, optim, lim, trainFile, trainaccFile, trainepochFile, validation_generator, valFile, valaccFile, pathm, i, modeltype):
    training_losses = []
    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        jI = 0
        totalBatches = 0
        scheduler.step()
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        lrFile.write('Epoch:'+' '+str(epoch)+' '+'LR:'+' '+str(scheduler.get_lr())+"\n")
        lrFile.write(str(scheduler.state_dict())+"\n")

        mb=0
        for trainim, trainmas in training_generator:
            mb+=1
            optim.zero_grad()
            images=trainim.to(device)
            masks=trainmas.to(device)
            outputs=model(images)
            if trainloss =='BCEWithLogitsLoss':
                l2_lambda=1e-4
                loss=nn.BCEWithLogitsLoss()
                output = loss(outputs, masks)
                l2_reg = sum(torch.norm(param, p=2) for param in model.parameters())
                output += l2_lambda * l2_reg
            output.backward()
            optim.step()
            
            batch_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0]
                       
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)
        trainFile.write(str(training_losses[epoch])+"\n")
        trainaccFile.write(str((jI/totalBatches).item())+"\n")
        trainepochFile.write(str(epoch)+"\n")
        print("Training Jaccard:",(jI/totalBatches).item()," (epoch:",epoch,")")
        lrFile.write("Training loss:"+str(training_losses[epoch])+"\n")
        lrFile.write("Training accuracy:"+str((jI/totalBatches).item())+"\n")
        
        
        torch.save(model.state_dict(), os.path.join(pathm, "iremmodel{}.pt".format(i)))
        validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype)
    torch.save(model.state_dict(), os.path.join(pathm, "Finaliremmodel{}.pt".format(i)))        
        
                
        
def validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype):
    jI = 0
    totalBatches = 0
    validation_losses = []
    
    
    if modeltype=='UNetV1':
        model = UNetV1(classes=1).to(device)
    elif modeltype=='UNetV2':
        model = UNetV2(classes=1).to(device)
    elif modeltype=='SegNet':
        model = SegNet(classes=1).to(device)
    elif modeltype=='DFANet':
        cfg=Config()
        model =  DFANet(cfg.ENCODER_CHANNEL_CFG,decoder_channel=64,num_classes=1).to(device)     
    elif modeltype=='DinkNet101':
        model =  DinkNet101(num_classes=1).to(device)
    elif modeltype=='DeepLabv3_plus':
        model = DeepLabv3_plus(num_classes=1, small=True, pretrained=True).to(device) 
    elif modeltype=='CamDUNet':
        model = CamDUNet().to(device)  
    elif modeltype=='R2U_Net':
        model = R2U_Net(img_ch=3,output_ch=1).to(device)        
    elif modeltype=='AttU_Net':
        model = AttU_Net(img_ch=3,output_ch=1).to(device)                
    elif modeltype=='R2AttU_Net':
        model = R2AttU_Net(img_ch=3,output_ch=1).to(device)    
    elif modeltype=='NestedUNet':
        model = NestedUNet(in_ch=3, out_ch=1).to(device)          
    elif modeltype=='DualNorm_Unet':
        model = DualNorm_Unet(n_channels=3, n_classes=1).to(device)        
    elif modeltype=='InceptionUNet':
        model = InceptionUNet(n_channels=3, n_classes=1, bilinear=True).to(device)        
    elif modeltype=='AttU_Net_with_scAG':
        model = AttU_Net_with_scAG(img_ch=3, output_ch=1,ratio=16).to(device)
    elif modeltype=='FSFNet':
        model = FSFNet(num_classes=1).to(device)      
    elif modeltype=='LMFFNet':
        model = LMFFNet(classes=1, block_1=3, block_2=8) .to(device) 
    elif modeltype=='LMFFNet2':
        model = LMFFNet2(classes=1, block_1=3, block_2=8) .to(device)
    elif modeltype=='LMFFNet3':
        model = LMFFNet3(classes=1, block_1=3, block_2=8) .to(device)         
    elif modeltype=='FASSDNet':
        model = FASSDNet(n_classes=1, alpha=2).to(device) 
    elif modeltype=='ENet':
        model = ENet(classes=1).to(device)  
    elif modeltype=='ELANet':  
        model = ELANet().to(device)  



    model.load_state_dict(torch.load(os.path.join(pathm, "iremmodel{}.pt".format(i))))
    model.eval()
    with torch.no_grad():
        val_losses = []
        for valim, valmas in validation_generator:
            #model.eval()
            images=valim.to(device)
            masks=valmas.to(device)
            outputs=model(images)
            if validationloss == 'BCEWithLogitsLoss':
                loss=nn.BCEWithLogitsLoss()
                output = loss(outputs, masks)
            val_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0] 
    dn=jI/totalBatches
    dni=dn.item()
    validation_loss = np.mean(val_losses)
    validation_losses.append(validation_loss)
    valFile.write(str(validation_losses[0])+"\n")
    valaccFile.write(str(dni)+"\n")
    print("Validation Jaccard:",dni)
    lrFile.write("Validation loss:"+str(validation_losses[0])+"\n")
    lrFile.write("Validation accuracy:"+str(dni)+"\n")
