import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
import torchvision.utils as vutils
import random
import torch.optim as optim
from torch.utils.data import DataLoader
## to start tensorboard: tensorboard --logdir=./airbus_scripts/summary --port=6006
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from rich.progress import track
from dtac.ClassDAE import *
from dtac.object_detection.yolo_model import YoloV1, YoloLoss
from dtac.object_detection.od_utils import *
from audio_DAE import DAE
CLEAN_SPEECH_SIGNAL_FOLDER_NAME = "clean_images/spectrograms_S02_P05"
NOISY_SPEECH_SIGNAL_FOLDER_NAME = "noisy_images"

def train_awa_vae(batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, weight_cross_penalty=0.1, 
                 device=0, lr=2e-4, seed=0, height=448, randpca=False, z_dim = 64):
    ### set paths
    model_type = "AE"
    cropped_image_size_h = 112
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("random seed: ", seed)
        print("randpca: ", randpca)

    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))

    files_dir =  './Data'
    
    # annots = []
    # for image in images:
    #     annot = image[:-4] + '.txt'
    #     annots.append(annot)
    # print("Total images: ",len(images))    
    # print("Total annots: ",len(annots))    
    # print("Total images: ",images[0])    
    # print("Total annots: ",annots[0])    
    # images = pd.Series(images, name='images')
    # annots = pd.Series(annots, name='annots')
    # df = pd.concat([images, annots], axis=1)
    # df = pd.DataFrame(df)
    # print(df.head())
    # print(f"resize to {height}x{height} then 448x448")
    # print("training set: ", file_parent_dir.split('/')[-2])
    p = 0.05
    print("p: ", p)
    transform_img = A.Compose(transforms=[
        A.Resize(width=height, height=height),
        # A.RandomResizedCrop(width=height, height=height),
        # A.Blur(p=p, blur_limit=(3, 7)), 
        # A.MedianBlur(p=p, blur_limit=(3, 7)), A.ToGray(p=p), 
        # A.CLAHE(p=p, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
        ToTensorV2(p=1.0)
    ])

    train_dataset = AudioDataset(
        files_dir=files_dir,
        # transform=transform_img,
        clean_image_folder = CLEAN_SPEECH_SIGNAL_FOLDER_NAME,
        noisy_image_folder = NOISY_SPEECH_SIGNAL_FOLDER_NAME
    )
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        # num_samples = 2,
    )

    ### load task model
    task_model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(device)
    # checkpoint = torch.load(task_model_path)
    # task_model.load_state_dict(checkpoint["state_dict"])
    # print("=> Loading checkpoint\n", "Train mAP:", checkpoint['Train mAP'], "\tTest mAP:", checkpoint['Test mAP'])
    task_model.eval()
    for param in task_model.parameters():
        param.requires_grad = False
    
    iou, conf = 0.5, 0.4
    print("iou: ", iou, "conf: ", conf)

    DVAE_awa = ResE2D1((3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h), int(z_dim/2), int(z_dim/2), 4, 1).to(device)
    print("ResBasedVAE Input shape", (3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h))
    DVAE_awa.train()
    optimizer = optim.Adam(DVAE_awa.parameters(), lr=lr)

    cur_iter = 0
    loss_fn = YoloLoss()
    for ep in range(num_epochs):
        ep_loss = []
        
        for batch_idx, data in enumerate(track(train_loader, description="Training: ")):
            clean_audio = data["clean_audio"]
            noisy_audio_1 = data["noisy_audio_1"]
            noisy_audio_2 = data["noisy_audio_2"]
            # ----------------------------------------------
            clean_audio = clean_audio.to(device)
            noisy_audio_1 = noisy_audio_1.to(device)
            noisy_audio_2 = noisy_audio_2.to(device)
            
            
            
            obs_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(noisy_audio_1, noisy_audio_2, clean_audio)
            
            
            obs_pred = torch.zeros_like(obs_).to(device) # 3x112x112

            
            
            # obs_112_0_255 = obs_pred.clip(0, 1) * 255.0 ##################### important: clip the value to 0-255
            # obs_pred_448_0_255 = F.interpolate(obs_112_0_255, size=(448, 448)) ### resize to 448x448
            # out_pred = task_model(obs_pred_448_0_255)
            # task_loss = loss_fn(out_pred, out)
            # loss = beta_task * task_loss + beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor
            # print(loss_rec ,kl1 , kl2, weight_cross_penalty , loss_cor)
            loss =  beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor
            ### check models' train/eval modes
            if (not DVAE_awa.training) or task_model.training:
                print(DVAE_awa.training, task_model.training)
                raise KeyError("Models' train/eval modes are not correct")
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            ep_loss.append(loss.item())
            cur_iter += 1

        ### print loss
        print("Epoch: {}, Loss: {}".format(ep, np.mean(ep_loss)))

        ### save model
        # if (ep + 1) % save_interval == 0 or (ep + 1) == 20 or ep == 0:
            ### test on train set
            # if "Joint" in vae_model:
            #     pred_boxes, target_boxes = get_bboxes_AE(
            #         train_loader, task_model, DVAE_awa, True, iou_threshold=iou, threshold=conf, device=device,
            #         cropped_audio_size_w=cropped_image_size, cropped_image_size_h=cropped_image_size
            #     )
            # else:
            #     pred_boxes, target_boxes = get_bboxes_AE(
            #         train_loader, task_model, DVAE_awa, False, iou_threshold=iou, threshold=conf, device=device,
            #         cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h
            #     )
            # train_mean_avg_prec = mean_average_precision(
            #     pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            # )
            # summary_writer.add_scalar(f'train_mean_avg_prec_{iou}_{conf}', train_mean_avg_prec, ep)    
            # print(train_mean_avg_prec, ep)

            ### test on test set
        #     if "Joint" in vae_model:
        #         pred_boxes, target_boxes = get_bboxes_AE(
        #             test_loader, task_model, DVAE_awa, True, iou_threshold=iou, threshold=conf, device=device,
        #             cropped_image_size_w=cropped_image_size, cropped_image_size_h=cropped_image_size
        #         )
        #     else:
        #         pred_boxes, target_boxes = get_bboxes_AE(
        #             test_loader, task_model, DVAE_awa, False, iou_threshold=iou, threshold=conf, device=device,
        #             cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h
        #         )
        #     test_mean_avg_prec = mean_average_precision(
        #         pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        #     )
        #     summary_writer.add_scalar(f'test_mean_avg_prec_{iou}_{conf}', test_mean_avg_prec, ep)
        #     print(test_mean_avg_prec, ep)

        #     torch.save(DVAE_awa.state_dict(), model_path + f'/DVAE_awa-{ep}.pth')  

        # ### export figure
        # if (ep + 1) % save_interval == 0 or ep == num_epochs - 1 or ep == 0:
        #     max_imgs = min(batch_size, 8)
        #     vutils.save_image(torch.cat([obs[:max_imgs], obs_pred[:max_imgs]], dim=0).data.cpu(),
        #         '{}/image_{}.jpg'.format(fig_dir, ep), nrow=8)

    return

if __name__ == "__main__":
    """        
    python train_od_awaAE.py --dataset airbus --device 6 -l 1e-4 -n 351 -r 0.0 -k 0.0 -t 0.1 -z 80 -bs 64 --seed 0 -corpen 0.0 -vae ResBasedVAE -wt 80 -ht 112 -p True
    python train_od_awaAE.py --dataset airbus --device 6 -l 1e-4 -n 351 -r 0.0 -k 0.0 -t 0.1 -z 80 -bs 64 --seed 0 -corpen 0.0 -vae JointResBasedVAE -wt 80 -ht 112
    """

    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str, help="dataset to train on: ['cifar10', 'airbus', 'PickAndPlace', 'gym_fetch']", default="airbus")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=351)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=80)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss", default=0.1)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence", default=0.1)
    parser.add_argument("-t", "--beta_task", type=float, help="beta coefficient for the task loss", default=0.1)
    parser.add_argument("-corpen", "--cross_penalty", type=float, help="cross-correlation penalty", default=0)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=0)
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device", default=0)
    parser.add_argument("-vae", "--vae_model", type=str, help="vae model: CNNBasedVAE or SVAE", default="ResBasedVAE")
    parser.add_argument("-wt", "--width", type=int, help="image width", default=80)
    parser.add_argument("-ht", "--height", type=int, help="image height", default=112)
    # parser.add_argument("-wt", "--width", type=int, help="image width", default=256)
    # parser.add_argument("-ht", "--height", type=int, help="image height", default=448)
    parser.add_argument("-p", "--randpca", type=bool, help="perform random pca when training", default=True)
    args = parser.parse_args()

    train_awa_vae(batch_size=args.batch_size, num_epochs=args.num_epochs, weight_cross_penalty=args.cross_penalty, 
                  beta_kl=args.beta_kl, beta_rec=args.beta_rec, device=args.device, lr=args.lr, seed=args.seed, height=args.height, randpca=args.randpca)


