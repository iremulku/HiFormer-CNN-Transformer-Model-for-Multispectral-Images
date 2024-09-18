import os
import torch 
import numpy as np
import torch.nn as nn
from F6_CROSSVAL import CrossVal
from F8_IMAGES import get_images
#from F8_IMAGES_RIT18 import get_images_rit18
from F3_DATASET import satellitedata
from torch.utils.data import DataLoader
from F9_UNET_V2_3 import UNetV2 
from F10_SEGNET_V1 import SegNet
from F20_DILATEDUNET import CamDUNet
from F21_GENERAL_UNET import R2U_Net, AttU_Net, R2AttU_Net
from F22_NESTEDUNET import NestedUNet
from F22_NESTEDUNET_IREM import NestedUNet2
from F23_DULANORM_UNET import DualNorm_Unet
from F24_INCEPTION_UNET import InceptionUNet
from IREM_INCEPTION import InceptionUNetIR
from F25_SCAG_UNET import AttU_Net_with_scAG
from F26_FSFNet import FSFNet
from F27_LMFFNet import LMFFNet
from LMFFNet_IREM import LMFFNet2
from LMFFNet_IREM3 import LMFFNet3
from F28_FASSDNet import FASSDNet
from F29_ENet import ENet
from F15_DFANET import DFANet
from F30_ELANet import ELANet
#from Swin_Unet import SwinTransformerSys
from F16_UNETFORMER2 import UNetFormer
from HiFormer import HiFormer
from HiFormer_irem import HiFormeri
from HiFormer_iremP import HiFormerP
from HiFormer_iremSE import HiFormerSE
import timm
from lora import LoRA_ViT
from base_vit import ViT
from seg_vit import SegWrapForViT
from F5_JACCARD2 import Jaccard2, Jaccard, JaccardAndF1
import matplotlib.pyplot as plt
from F11_SEGPLOT2 import segplot
from F14_DEEPLABV3PLUS_V4_xception import DeepLabv3_plus

import warnings

class Config(object):
    NAME= "dfaNet"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]


warnings.filterwarnings("ignore")
createFigures = False
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 

#main
fout = open("iremf1.txt", "w")

with open('irem-input-rit18.txt') as f:
    contents = f.readlines()
prevline = ""
prevInputType = ""
for line in contents:        
    if line[0]=='D':
        # the folder
        modelPath = line.replace("\n","")
        liste = os.listdir(modelPath)
        # print(liste)
        
        # the model name (does it exist)
        FinalExists = False;
        for afile in liste:                
            if (afile[-3:-1] + afile[-1]) == ".pt":
                if afile[0:5] == "Final":
                    FinalExists = True;
                    break
        # if a final***.pt model exists get the model number in the end
        if FinalExists:
            if line[-3]!='l':
                modelName = "Finaliremmodel" + line[-3:-1] + ".pt" 
            else:
                modelName = "Finaliremmodel" + line[-2] + ".pt"
                
        else:
            modelName = prevline + ".pt"
        print(modelName)
        
        # find the log file
        logfile = ""
        for afile in liste:                
            if (afile[-3:-1] + afile[-1]) == "txt":
                logfile = afile 
                break
        with open(modelPath + "/" + logfile) as log:
            logs = log.readlines()
            # fold number 
            foldNo = int(logs[4][-2:-1])
            # the input modality
            inputType = logs[18][14:-1]
            # the model
            modelType = logs[21][0:-2]
            print( modelType)
            # print( modelType)
            # if logs[22][3:6] ==  "xce":
            #     modelType = "DeepLabv3_plusX"
            # if logs[22][3:6] ==  "res":
            #     if logs[216][0:6] == "Epoch:":
            #         modelType = "DeepLabv3_plus34"
            #     else:
            #         modelType = "DeepLabv3_plus101"
           
            #if ((inputType=='all20Ch') | (modelType=='SegNet') | (inputType=='NDVI_NDMI_NDWI_WRI_ARVI_SAVI')):
            # if (inputType=='RGBs'):
            #     continue
            
           
            
            figuresPath = "results/" + modelType + '_' + inputType + '_' + str(foldNo)
            if createFigures & ~os.path.isdir(figuresPath) :
                os.mkdir(figuresPath)
           
           
           
            ### now run the Jaccard ###                
            #first construct the model

            if modelType=='UNetV2':
                model = UNetV2(classes=1).to(device)
            if modelType=='SegNet':
                model = SegNet(classes=1).to(device) 
            if modelType=='DeepLabv3_plus':
                model = DeepLabv3_plus(num_classes=1, small=True, pretrained=True).to(device)                
            if modelType=='CamDUNet':
                model = CamDUNet().to(device)
            if modelType=='R2U_Net':
                model = R2U_Net(img_ch=3,output_ch=1).to(device)        
            if modelType=='AttU_Net':
                model = AttU_Net(img_ch=3,output_ch=1).to(device)                
            if modelType=='R2AttU_Net':
                model = R2AttU_Net(img_ch=3,output_ch=1).to(device)  
            if modelType=='NestedUNet':
                model = NestedUNet(in_ch=3, out_ch=1).to(device) 
            if modelType=='NestedUNet2':
                model = NestedUNet2(in_ch=3, out_ch=1).to(device)               
            if modelType=='DualNorm_Unet':
                model = DualNorm_Unet(n_channels=3, n_classes=1).to(device)               
            if modelType=='InceptionUNet':
                model = InceptionUNet(n_channels=3, n_classes=1, bilinear=True).to(device)   
            if modelType=='InceptionUNetIR':
                model = InceptionUNetIR(in_ch=3, out_ch=1).to(device)                  
            if modelType=='AttU_Net_with_scAG':
                model = AttU_Net_with_scAG(img_ch=3, output_ch=1,ratio=16).to(device)     
            if modelType=='FSFNet':
                model = FSFNet(num_classes=1).to(device) 
            if modelType=='LMFFNet':
                model = LMFFNet(classes=1, block_1=3, block_2=8) .to(device) 
            if modelType=='LMFFNet2':
                model = LMFFNet2(classes=1, block_1=3, block_2=8) .to(device) 
            if modelType=='LMFFNet3':
                model = LMFFNet3(classes=1, block_1=3, block_2=8) .to(device)                 
            if modelType=='FASSDNet':
                model = FASSDNet(n_classes=1, alpha=2).to(device) 
            if modelType=='ENet':
                model = ENet(classes=1).to(device)                   
            if modelType=='DFANet':
                cfg=Config()
                model =  DFANet(cfg.ENCODER_CHANNEL_CFG,decoder_channel=64,num_classes=1).to(device)                
            # if modelType=='SwinTransformerSys':    
            #     model = SwinTransformerSys(img_size=224, embed_dim=96, in_chans=3, num_classes=1, patch_size=4,
            #                depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
            #                window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            #                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            #                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            #                use_checkpoint=False, final_upsample="expand_first").to(device) 
            if modelType=='ELANet':  
                model = ELANet().to(device)   
            if modelType == 'UNetFormer':
                model = UNetFormer(decode_channels=64, #64
                     dropout=0.1,
                     backbone_name='swsl_resnet18', #resnet18
                     pretrained=False, # was true
                     window_size=4, #8 
                     num_classes=1).to(device)                
            if modelType == 'HiFormer':
                model = HiFormer().to(device)
            if modelType == 'HiFormeri':
                model = HiFormeri().to(device)
            if modelType == 'HiFormerSE':
                model = HiFormerSE().to(device)                  
            if modelType == 'SegWrapForViT1':
                model1 = ViT('B_16_imagenet1k')
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=16, dim=768, n_classes=1).to(device)                
            if modelType == 'SegWrapForViT2':
                model1 = ViT('B_16_imagenet1k')
                model = SegWrapForViT(vit_model=model1, image_size=224,
                                            patches=16, dim=768, n_classes=1).to(device)
            if modelType == 'SegWrapForViT':
                model1 = ViT('L_16_imagenet1k')
                # LoRA_ViT3
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=16, dim=1024, n_classes=1).to(device)
            if modelType == 'SegWrapForViT4':
                model1 = ViT('L_16_imagenet1k')
                # LoRA_ViT4
                model = SegWrapForViT(vit_model=model1, image_size=224,
                                            patches=16, dim=1024, n_classes=1).to(device)       
            if modelType == 'SegWrapForViT5':
                model1 = ViT('B_16')
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=16, dim=768, n_classes=1).to(device) 
            if modelType == 'SegWrapForViT6':
                model1 = ViT('B_32_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=32, dim=768, n_classes=1).to(device)
            if modelType == 'SegWrapForViT7':
                model1 = ViT('B_32_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
                # model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=model1, image_size=224,
                                            patches=32, dim=768, n_classes=1).to(device)        
            if modelType == 'SegWrapForViT8':
                model1 = ViT('L_32_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=32, dim=1024, n_classes=1).to(device)
            if modelType == 'SegWrapForViT9':
                model1 = ViT('L_32_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
                # model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=model1, image_size=224,
                                            patches=32, dim=1024, n_classes=1).to(device)          
                
                
                
                
        
# change  SegWrapForViT and SegWrapForViT2 for models with or without lora     
        # and load the model 
            print(modelPath + "/" + modelName)             
            model.load_state_dict(torch.load(modelPath + "/" + modelName))
            model.eval()
            
            # load input (for DSTL and RIT18) 
            
            # tsind,trind,vlind = CrossVal(5985,foldNo,5);
            tsind,trind,vlind = CrossVal(1778,foldNo,2);
            # if prevInputType!=inputType:
            #     if inputType!='all20Ch':
                #    input_images, target_masks, trMean10, trMean11, trMean12, names = get_images_nir(1778, foldNo, 2, tsind, trind, vlind, inputType)
                #else:
            input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images(1778, foldNo, 2, tsind, trind, vlind, inputType)
            #input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images(5985, foldNo, 5, tsind, trind, vlind, inputType)
            #prevInputType = inputType
            #dataloader for testset
            params = {'batch_size': 1, 'shuffle': False}    
            test_set = satellitedata(input_images[tsind], target_masks[tsind])
            test_generator = DataLoader(test_set, **params)
            
            f1All = np.empty(test_generator.dataset.images.shape[0],dtype='float')
            jcrdsAll = np.empty(test_generator.dataset.images.shape[0],dtype='float')

            with torch.no_grad():
                ts = 0;
                for testim, testmas in test_generator:
                    images=testim.to(device)
                    masks=testmas.to(device)
                    outputs = model(images) 
 

                    
                    f1 = JaccardAndF1(torch.reshape(masks,(224*224,1)),torch.reshape(outputs,(224*224,1)))                                    
                    jcrd = Jaccard2(torch.reshape(masks,(224*224,1)),torch.reshape(outputs,(224*224,1)))
                    jcrdsAll[ts] = jcrd.to('cpu').numpy()[0]
                    f1All[ts] = f1.to('cpu').numpy()[0]

                    
                    # create all test figures for this fold
                    if createFigures:
                        fimage=images[0].permute(1, 2, 0)
                        fimage[:,:,0]=(images[0][0,:,:])
                        fimage[:,:,1]=(images[0][1,:,:])
                        fimage[:,:,2]=(images[0][2,:,:])
                        fimage=fimage.cpu().numpy()
                        foutput=outputs[0].permute(1, 2, 0)
                        foutput=foutput.cpu().numpy()
                        fmask=masks[0].permute(1, 2, 0)
                        fmask=fmask.cpu().numpy()
                        segplot(figuresPath, 224, fimage, foutput, fmask,  trMeanR, trMeanG, trMeanB, ts)
                    
                    ts = ts+1;      
            
            
            #print(modelType + ", " + inputType + ", Mean Precision:", mean_precision, "±" , std_precision)
            print(modelType + ", " + inputType + ", f1: ", f1All.mean() , "±" , f1All.std())
            print(modelType + ", " + inputType + ", Jaccard: ", jcrdsAll.mean() , "±" , jcrdsAll.std())
            #fout.write(modelType + '\t' + inputType + '\t' + str(f1All.mean()) + '\t' + str(f1All.std()) + '\t' + modelType + '\n')
    else:
        prevline = line[0:-1]
fout.close()