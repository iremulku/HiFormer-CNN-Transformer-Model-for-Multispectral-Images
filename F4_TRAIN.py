from __future__ import print_function
import os
import torch 
import torch.nn as nn
import numpy as np
from F5_JACCARD2 import Jaccard2
from F1_UNET_V1_1 import UNetV1
#from F9_UNET_V2_4 import UNetV2
from F9_UNET_V2_3 import UNetV2
from F10_SEGNET_V1 import SegNet
from F15_DFANET import DFANet 
from F12_DLINKNET_V3 import DinkNet101
from F12_DLINKNET_V2 import DinkNet34
from F20_DILATEDUNET import CamDUNet
from F21_GENERAL_UNET import R2U_Net, AttU_Net, R2AttU_Net
from F22_NESTEDUNET import NestedUNet
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
from F30_ELANet import ELANet
from F16_UNETFORMER2 import UNetFormer
from F32_SEGFORMER import Segformer
from HiFormer import HiFormer
from HiFormer_irem import HiFormeri
from HiFormer_iremP import HiFormerP
from HiFormer_iremSE import HiFormerSE
import timm
from lora import LoRA_ViT
from base_vit import ViT
from seg_vit import SegWrapForViT
#from timm.models.vision_transformer import VisionTransformer as timm_ViT
#from Swin_Unet import SwinTransformerSys
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
    t_layer_i = 0
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
            if mb!=101:
                optim.zero_grad()
                images=trainim.to(device)
                masks=trainmas.to(device)
                outputs=model(images)
                #print("output:", outputs.shape)
                #print("masks:", masks.shape)
                if trainloss =='BCEWithLogitsLoss':
                    loss=nn.BCEWithLogitsLoss()
                    output = loss(outputs, masks)            
                output.backward()
                
                # for name, param in model.named_parameters():  
                #     if name == "vit.lora_vit.transformer.blocks.2.attn.proj_q.w_b.weight":
                #         print("Gradients of w_b_linear:", param.grad)
                #     elif name =="vit.lora_vit.transformer.blocks.2.attn.proj_q.w.weight":
                #         print("Gradients of w_q_linear:", param.grad)
                #     elif name =="vit.lora_vit.transformer.blocks.2.attn.proj_q.w_a.weight":
                #         print("Gradients of w_a_linear:", param.grad)

                optim.step()
            else:
                pass

            
            batch_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                thisJac = Jaccard2(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0]
                #print("accuracy", jI)
                       
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

    
    lora_layer_params = []
    for name, param in model.named_parameters():
        if '_LoRALayer' in name:
            lora_layer_params.append((name, param))
    for name, param in lora_layer_params:
        print(f"Parameter name: {name}")
        print(param.data)            
                
        
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
    elif modeltype=='DinkNet34':
        model = DinkNet34(num_classes=1, num_channels=3).to(device)             
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
    elif modeltype=='InceptionUNetIR':
        model = InceptionUNetIR(in_ch=3, out_ch=1).to(device)  
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
    elif modeltype == 'UNetFormer':
        model = UNetFormer(decode_channels=64, #64
                 dropout=0.1,
                 backbone_name='swsl_resnet18', #resnet18
                 pretrained=False, # was true
                 window_size=4, #8 
                 num_classes=1).to(device)  
    elif modeltype == 'HiFormer':
        model = HiFormer().to(device)
    elif modeltype == 'HiFormeri':
        model = HiFormeri().to(device)
    elif modeltype == 'HiFormerSE':
        model = HiFormerSE().to(device)          
    elif modeltype == 'LoRA_ViT':
        model1 = ViT('B_16_imagenet1k')
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)    
    # elif modeltype == 'LoRA_ViT':
    #     model1 = ViT('B_16_imagenet1k')
    #     model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
    #     #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))  
    #     lora_model = LoRA_ViT(model1, r=4).to(device)
    #     model = SegWrapForViT(vit_model=lora_model, image_size=224,
    #                                 patches=16, dim=768, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT2':
        model1 = ViT('B_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        #model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)          
    elif modeltype == 'LoRA_ViT3':
        model1 = ViT('L_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=1024, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT4':
        model1 = ViT('L_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=16, dim=1024, n_classes=1).to(device)        
    elif modeltype == 'LoRA_ViT5':
        model1 = ViT('B_16')
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)            
    elif modeltype == 'LoRA_ViT6':
        model1 = ViT('B_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=32, dim=768, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT7':
        model1 = ViT('B_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=32, dim=768, n_classes=1).to(device)  
    elif modeltype == 'LoRA_ViT8':
        model1 = ViT('L_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=32, dim=1024, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT9':
        model1 = ViT('L_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=32, dim=1024, n_classes=1).to(device)             









        
        

        # model.save_lora_parameters('mytask.lora.safetensors') # save
        # model.load_lora_parameters('mytask.lora.safetensors') # load


    model.load_state_dict(torch.load(os.path.join(pathm, "iremmodel{}.pt".format(i))))
    model.eval()
    
    
    lora_layer_params = []
    for name, param in model.named_parameters():
        if '_LoRALayer' in name:
            lora_layer_params.append((name, param))
    for name, param in lora_layer_params:
        print(f"Parameter name: {name}")
        print(param.data)  
        
        
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
                thisJac = Jaccard2(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                #print("maskeler", torch.reshape(masks,(batchLoad,1)))
                #print("tahminler", torch.reshape(outputs,(batchLoad,1)))
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
