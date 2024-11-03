import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
import torchvision.utils as vutils
import random
import csv
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
CLEAN_SPEECH_SIGNAL_FOLDER_NAME = "clean_images/spectrograms_S02_P08"
NOISY_SPEECH_SIGNAL_FOLDER_NAME = "noisy_images"



def save_checkpoint(model, optimizer, epoch, loss, model_path, filename):
    """Save model checkpoint with additional training info"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(checkpoint, os.path.join(model_path, filename))

def load_checkpoint(model, optimizer, model_path, filename):
    """Load model checkpoint and return epoch and loss"""
    checkpoint_path = os.path.join(model_path, filename)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    return 0, None

def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=100, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                 device=0, save_interval=30, lr=2e-4, seed=0, vae_model="CNNBasedVAE", width=448, height=448, randpca=False,):
    ### set paths
    model_type = "AE"

    LOG_DIR = f'./summary/{dataset}_{z_dim}_randPCA_{model_type}_{vae_model}'
    fig_dir = f'./figures/{dataset}_{z_dim}_randPCA_{model_type}_{vae_model}'
    model_path = f'./models/{dataset}_{z_dim}_randPCA_{model_type}_{vae_model}'
    files_dir =  './Data'
    
    if not randpca:
        LOG_DIR = LOG_DIR.replace("randPCA", "NoPCA")
        fig_dir = fig_dir.replace("randPCA", "NoPCA")
        model_path = model_path.replace("randPCA", "NoPCA")
    summary_writer = SummaryWriter(os.path.join(LOG_DIR, 'tb'))

    ### Set the random seed
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

    train_dataset = ImagesDataset(
        files_dir=files_dir,
        transform=transform_img,
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
    )
    test_dataset = ImagesDataset(
        files_dir=files_dir,
        transform=transform_img,
        clean_image_folder = CLEAN_SPEECH_SIGNAL_FOLDER_NAME,
        noisy_image_folder = NOISY_SPEECH_SIGNAL_FOLDER_NAME,
        folder_name="Test"
    )
    g = torch.Generator()
    g.manual_seed(0)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    cropped_image_size_w = width
    cropped_image_size_h = height
    cropped_image_size = height

 
    test_transform_img = A.Compose([
        A.Resize(width=height, height=height),
        ToTensorV2(p=1.0)
    ])
    
   

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

    ### distributed models
    if vae_model == "CNNBasedVAE":
        DVAE_awa = E2D1((3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_w, cropped_image_size_h), int(z_dim/2), int(z_dim/2), 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("CNNBasedVAE Input shape", (3, cropped_image_size_w, cropped_image_size_h))
    elif vae_model == "ResBasedVAE":
        DVAE_awa = ResE2D1((3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h), int(z_dim/2), int(z_dim/2), 4, 1).to(device)
        print("ResBasedVAE Input shape", (3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h))
    ### Joint models
    elif vae_model == "JointCNNBasedVAE":
        DVAE_awa = E1D1((6, cropped_image_size, cropped_image_size), z_dim, 4, int(128/(seed+1)), 2, 128).to(device)
        print("JointCNNBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "JointResBasedVAE":
        DVAE_awa = ResE1D1((6, cropped_image_size, cropped_image_size), z_dim, 4, 1).to(device)
        print("JointResBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "SepResBasedVAE":
        DVAE_awa = ResE2D2((3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h), int(z_dim/2), int(z_dim/2), 4, 1).to(device) ### 4, 1 to balance # of paras
        print("SepResBasedVAE Input shape", (3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h))
    else:
        raise NotImplementedError
    
    # _ = ResE2D2((3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h), int(z_dim/2), int(z_dim/2), 4, 1).to(device)
    # print("ResE2D1 trainable parameters: ", count_parameters(DVAE_awa))
    # print("ResE2D2 trainable parameters: ", count_parameters(_))
    # exit(0)


    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


    DVAE_awa.train()
    optimizer = optim.Adam(DVAE_awa.parameters(), lr=lr)
    # Try to load the latest checkpoint
    latest_checkpoint = 'latest_checkpoint.pth'
    start_epoch, last_loss = load_checkpoint(DVAE_awa, optimizer, model_path, latest_checkpoint)
    
    if last_loss is not None:
        print(f"Resuming training from epoch {start_epoch} with loss {last_loss}")
    else:
        start_epoch = 0
        print("Starting training from scratch")


    cur_iter = 0
    loss_fn = YoloLoss()
    epoch_data = []

    
    for ep in range(num_epochs):
        ep_loss = []
        total_k1 = 0
        total_k2 = 0
        total_rec_loss = 0
        total_loss_cor = 0
        total_spec_total_loss = 0
        total_spec_total_loss = 0
        total_psnr = 0
        total_freq_weighted_loss = 0
        total_smoothness_loss = 0
        
        for batch_idx, data in enumerate(track(train_loader, description="Training: ")):
            # obs_112_0_255, out = obs.to(device), out.to(device)
            # obs = obs_112_0_255 / 255.0
            # o1_batch = torch.zeros(obs.shape[0], obs.shape[1], cropped_image_size_h, cropped_image_size_h).to(device)
            # o2_batch = torch.zeros(obs.shape[0], obs.shape[1], cropped_image_size_h, cropped_image_size_h).to(device)
            # o1_batch[:, :, :cropped_image_size_w, :cropped_image_size_h] = obs[:, :, :cropped_image_size_w, :cropped_image_size_h]
            # o2_batch[:, :, cropped_image_size_w-20:, :cropped_image_size_h] = obs[:, :, cropped_image_size_w-20:, :cropped_image_size_h]
            clean_image = data["clean_image"]
            noisy_image_1 = data["noisy_image_1"]
            noisy_image_2 = data["noisy_image_2"]
            # ----------------------------------------------
            clean_image = clean_image.to(device)
            noisy_image_1 = noisy_image_1.to(device)
            noisy_image_2 = noisy_image_2.to(device)
            
            clean_image = clean_image / 255.0
            noisy_image_1 = noisy_image_1 / 255.0
            noisy_image_2 = noisy_image_2 / 255.0
            
            # Correct tensor concatenation along dimension 1 (channel dimension)
            noisy_images_concat = torch.cat((noisy_image_1, noisy_image_2), dim=1)
            
            obs_, loss_rec, kl1, kl2, loss_cor,spec_total_loss, spec_loss_dict, psnr, dimension_info = DVAE_awa(noisy_images_concat)
            
            # if "Joint" in vae_model:
            #     obs_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(torch.cat((o1_batch, o2_batch), dim=1))
            # elif "Sep" in vae_model:
            #     obs_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch)
            # elif "Joint" not in vae_model:
            #     obs_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch, random_bottle_neck=randpca)
            
            ### post processing 6 channels to 3 channels
            obs_pred = torch.zeros_like(obs_).to(device) # 3x112x112

            
            
            # obs_112_0_255 = obs_pred.clip(0, 1) * 255.0 ##################### important: clip the value to 0-255
            # obs_pred_448_0_255 = F.interpolate(obs_112_0_255, size=(448, 448)) ### resize to 448x448
            # out_pred = task_model(obs_pred_448_0_255)
            # task_loss = loss_fn(out_pred, out)
            # loss = beta_task * task_loss + beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor
            # print(loss_rec ,kl1 , kl2, weight_cross_penalty , loss_cor)
            loss =  beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor  + (spec_total_loss)
            ### check models' train/eval modes
            if (not DVAE_awa.training) or task_model.training:
                print(DVAE_awa.training, task_model.training)
                raise KeyError("Models' train/eval modes are not correct")
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### log tensorboard
            summary_writer.add_scalar('loss', loss, cur_iter)
            # summary_writer.add_scalar('task', task_loss, cur_iter)
            summary_writer.add_scalar('Rec', loss_rec, cur_iter)
            summary_writer.add_scalar('kl1/var', kl1, cur_iter)
            summary_writer.add_scalar('kl2/invar', kl2, cur_iter)
            summary_writer.add_scalar('cov', loss_cor, cur_iter)
            summary_writer.add_scalar('PSNR', psnr, cur_iter)

            ep_loss.append(loss.item())
            cur_iter += 1
            total_rec_loss =+ loss_rec
            total_loss_cor =+ loss_cor
            total_spec_total_loss =+ spec_total_loss
            total_psnr =+ psnr
            total_freq_weighted_loss =+ spec_loss_dict["freq_weighted_loss"]
            total_smoothness_loss =+ spec_loss_dict["smoothness_loss"]
        # print(total_rec_loss, train_loader, total_rec_loss/len(train_loader))
        epoch_data.append({
            "epoch": ep + 1,
            "model_name": vae_model,
            "kl1": total_k1/len(train_loader),
            "kl2": total_k1/len(train_loader),
            "loss_rec": (total_rec_loss/len(train_loader)).item(),
            "loss_cor": (total_loss_cor/len(train_loader)),
            "spec_total_loss": (total_spec_total_loss/len(train_loader)).item(),
            "psnr": (total_psnr/len(train_loader)).item(),
            "total_smoothness_loss": (total_smoothness_loss/len(train_loader)).item(),  # Dictionary for each epoch
            "total_freq_weighted_loss": (total_freq_weighted_loss/len(train_loader)).item(),  # Dictionary for each epoch
            **dimension_info,
        })

        # Define the CSV file path

        csv_file_path = f'{vae_model}_training_results.csv'
        # Writing data to CSV
        with open(csv_file_path, mode='w', newline='') as csv_file:
            fieldnames = ["epoch", "model_name", "kl1", "kl2", "loss_rec", "loss_cor", "spec_total_loss", "psnr", "total_smoothness_loss", "total_freq_weighted_loss"]
            fieldnames.extend(dimension_info.keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for data in epoch_data:
                writer.writerow(data)

        

        epoch_loss = np.mean(ep_loss)
        print(f"Epoch: {ep}, Loss: {epoch_loss}")
        


        
        # Save checkpoint after every epoch
        save_checkpoint(
            DVAE_awa, 
            optimizer, 
            ep, 
            epoch_loss, 
            model_path, 
            f'checkpoint_epoch_{ep}.pth'
        )
        
        # Also save as latest checkpoint (overwrite)
        save_checkpoint(
            DVAE_awa, 
            optimizer, 
            ep, 
            epoch_loss, 
            model_path, 
            'latest_checkpoint.pth'
        )

        # if (ep + 1) % save_interval == 0 or ep == num_epochs - 1:
        #     max_imgs = min(batch_size, 8)
            
        #     # Now both noisy_images_concat and obs_ should have same number of channels
        #     vutils.save_image(
        #         torch.cat([
        #             noisy_images_concat[:max_imgs],  # Original concatenated input
        #             obs_[:max_imgs]                  # Reconstructed concatenated output
        #         ], dim=0).data.cpu(),
        #         '{}/image_{}.jpg'.format(fig_dir, ep), 
        #         nrow=8
        #     )

    print("Training completed!")

        # After training loop
    epoch_data = []
    # Initialize variables for test metrics
    total_test_loss = 0
    total_test_psnr = 0
    num_test_batches = len(test_loader)

    # Testing loop
    with torch.no_grad():
        total_k1 = 0
        total_k2 = 0
        total_rec_loss = 0
        total_loss_cor = 0
        total_spec_total_loss = 0
        total_spec_total_loss = 0
        total_psnr = 0
        total_freq_weighted_loss = 0
        total_smoothness_loss = 0
        for batch_idx, data in enumerate(track(test_loader, description="Testing: ")):
            clean_image = data["clean_image"].to(device) / 255.0
            noisy_image_1 = data["noisy_image_1"].to(device) / 255.0
            noisy_image_2 = data["noisy_image_2"].to(device) / 255.0
            noisy_images_concat1 = torch.cat((noisy_image_1, noisy_image_2), dim=1)
            
            obs_, loss_rec, kl1, kl2, loss_cor, spec_total_loss, spec_loss_dict, psnr, dimension_info = DVAE_awa(noisy_images_concat1)
            # Calculate test loss and PSNR (you might have a specific way to compute this based on your task)
            test_loss =  beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor  + (spec_total_loss)

            total_test_loss += test_loss.item()
            total_test_psnr += psnr.item()
            
            total_rec_loss =+ loss_rec
            total_loss_cor =+ loss_cor
            total_spec_total_loss =+ spec_total_loss
            total_psnr =+ psnr
            total_freq_weighted_loss =+ spec_loss_dict["freq_weighted_loss"]
            total_smoothness_loss =+ spec_loss_dict["smoothness_loss"]
    # Average the test metrics
    average_test_loss = total_test_loss / num_test_batches
    average_test_psnr = total_test_psnr / num_test_batches

    # Now, append the test metrics to epoch_data
    epoch_data.append({
        "epoch": ep + 1,
        "model_name": vae_model,
        "kl1": total_k1 / len(train_loader),
        "kl2": total_k1 / len(train_loader),
        "loss_rec": (total_rec_loss / len(train_loader)).item(),
        "loss_cor": (total_loss_cor / len(train_loader)),
        "spec_total_loss": (total_spec_total_loss / len(train_loader)).item(),
        "total_smoothness_loss": (total_smoothness_loss / len(train_loader)).item(),
        "total_freq_weighted_loss": (total_freq_weighted_loss / len(train_loader)).item(),
        "test_loss": average_test_loss,  # Add test loss
        "test_psnr": average_test_psnr,  # Add test PSNR
        **dimension_info,
    })

    # Define the CSV file path
    csv_file_path = f'{vae_model}_testing_results.csv'

    # Writing data to CSV
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = [
            "epoch", "model_name", "kl1", "kl2", "loss_rec", "loss_cor",
            "spec_total_loss", "psnr", "total_smoothness_loss", 
            "total_freq_weighted_loss", "test_loss", "test_psnr"
        ]
        fieldnames.extend(dimension_info.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in epoch_data:
            writer.writerow(data)

    return

if __name__ == "__main__":
    """        
    python train_od_awaAE.py --dataset airbus --device 6 -l 1e-4 -n 351 -r 0.0 -k 0.0 -t 0.1 -z 80 -bs 64 --seed 0 -corpen 0.0 -vae ResBasedVAE -wt 80 -ht 112 -p True
    python train_od_awaAE.py --dataset airbus --device 6 -l 1e-4 -n 351 -r 0.0 -k 0.0 -t 0.1 -z 80 -bs 64 --seed 0 -corpen 0.0 -vae JointResBasedVAE -wt 80 -ht 112
    """

    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str, help="dataset to train on: ['cifar10', 'airbus', 'PickAndPlace', 'gym_fetch']", default="chime6")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=100)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=20)
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
    parser.add_argument("-p", "--randpca", type=bool, help="perform random pca when training", default=False)
    args = parser.parse_args()

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, weight_cross_penalty=args.cross_penalty, 
                  beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, device=args.device, save_interval=2, lr=args.lr, seed=args.seed,
                  vae_model="JointCNNBasedVAE", width=args.width, height=args.height, randpca=False)



# python train_od_awaAE.py --dataset airbus --device 0 -l 1e-4 -n 351 -r 0.0 -k 0.0 -t 0.1 -z 80 -bs 8 --seed 0 -corpen 0.0 -vae ResBasedVAE -wt 80 -ht 112 -p True