import argparse
import os
import numpy as np
import time
from modeling.coanet import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid #, save_image
from dataloaders import make_data_loader
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses
from tqdm import tqdm
import re
import random
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader





os.environ['CUDA_VISIBLE_DEVICES'] = ''


class RandomSamplerWithReplacement(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


class Args:
    def __init__(self):
        self.no_cuda = False
        self.batch_size = 1
        # self.ckpt = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\CoANet-DeepGlobe.pth.tar'
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        self.gpu_ids = '0'#'' will be empty string and will not be using any gpu 
        self.dataset = '7b52009b64fd0a2a49e6d8a939753077792b0554'
        self.base_size = 280
        self.crop_size = 280
        # self.base_size = 1280
        # self.crop_size = 1280
        self.sync_bn = False
        self.freeze_bn = False

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize([280, 280])
    im.save(filename)

def main():
    
    args = Args()

    best_acc = 0.0
    best_acc_class = 0.0
    best_mIoU = 0.0
    best_IoU = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0



    base_path = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\out_imgs_1300'
    dataset_name = args.dataset  # Replace 'your_dataset_name' with the actual dataset name
    result = 'result8'  # Replace 'result1' with the actual result1 value

    # Join the components to create the new path
    out_path = os.path.join(base_path, dataset_name, result)

    # out_path = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\out_imgs_1300\size280_gpu0_result4'
    # out_path = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\out_imgs_1300\size280_gpu0_result3'
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    if args.gpu_ids:  # Check if gpu_ids is not an empty string
        print("Using GPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    else:
        print("Using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    train_loader, val_loader, test_loader, nclass = make_data_loader(args, num_workers=0, pin_memory=False)

    model = CoANet(num_classes=nclass,
                    backbone='resnet',
                    output_stride=8,
                    sync_bn=False,
                    freeze_bn=False)
    model.eval()



    if args.gpu_ids:  # Check if gpu_ids is not an empty string
        # ckpt = torch.load(r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\run\model\model_best.pth.tar')
        ckpt = torch.load(r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\run\model\model_best.pth.tar')
    else:
        # ckpt = torch.load(r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\run\model\model_best.pth.tar', map_location=torch.device('cpu'))
        ckpt = torch.load(r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\run\model\model_best.pth.tar', map_location=torch.device('cpu'))
    
   
    
    # print(ckpt['state_dict'].keys())
    # print(ckpt['state_dict'])


    model.load_state_dict(ckpt['state_dict'])

    # ckpt = torch.load(r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\CoANet-DeepGlobe.pth.tar', map_location=torch.device('cpu'))
    # model.load_state_dict(ckpt['state_dict'])

    evaluator = Evaluator(2)
    evaluator.reset()
    # processed_image_count = 0  # Counter for processed images

    # tbar = tqdm(test_loader, desc='\r')
    # print("tbar length:", len(test_loader))

    # # Extract the starting frame number from the first image filename
    # first_image_filename = os.path.basename(test_loader.dataset.images[0])
    # frame_number_match = re.search(r'frame(\d+)_sat', first_image_filename)
    # if frame_number_match:
    #     frame_number_str = frame_number_match.group(1)  # Extract the captured group
    #     frame_number = int(frame_number_str)
    #     print(frame_number)
    # else:
    #     frame_number = 0  # Use the loop index as fallback


    frame_metrics = []  # List to store frame metrics (frame_number, metrics_dict)

    # Shuffle the test_loader
    # tbar = list(test_loader)
    # random.shuffle(tbar)
    # tbar = tqdm(test_loader, desc='\r')
    # print("tbar length:", len(test_loader))
    # Create a custom data sampler to shuffle the test_loader
    test_sampler = RandomSamplerWithReplacement(test_loader.dataset)

    # Create DataLoader with the custom data sampler
    shuffled_test_loader = DataLoader(dataset=test_loader.dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=0, pin_memory=False)

    # Create a tqdm object for the shuffled test_loader with a progress bar
    tbar = tqdm(shuffled_test_loader, desc='\r')
    print("tbar length:", len(shuffled_test_loader))




    for i, sample in enumerate(tbar):
        # img_name = sample[1][0].split('.')[0]

        # # Extract the frame number using regular expression
        # frame_number_match = re.search(r'frame(\d+)_sat', img_name)
        # if frame_number_match:
        #     frame_number_str = frame_number_match.group(1)  # Extract the captured group
        #     frame_number = int(frame_number_str)
        # else:
        #     frame_number = i  # Use the loop index as fallback

        # print("Frame number:", frame_number)  # Print to verify

        # Extract the frame number from the filename
        img_name = sample[1][0].split('.')[0]
        print("img_name : ",img_name)

        # frame_number_match = re.search(r'frame(\d+)_sat', img_name)
        # print("frame_number_match : ",frame_number_match)

        if img_name:
            frame_number = img_name  # Extract the captured group
            print("Frame number : ",frame_number)
        else:
            frame_number = i  # Use the loop index as a fallback

        image, target = sample[0]['image'], sample[0]['label']
        image = image.cpu().numpy()
        image1 = image[:, :, ::-1, :]
        image2 = image[:, :, :, ::-1]
        image3 = image[:, :, ::-1, ::-1]
        image = np.concatenate((image, image1, image2, image3), axis=0)
        image = torch.from_numpy(image).float()

        img_name = sample[1][0].split('.')[0]
        
        
        output, out_connect, out_connect_d1 = model(image)

        out_connect_full = []
        out_connect = out_connect.data.cpu().numpy()
        out_connect_full.append(out_connect[0, ...])
        out_connect_full.append(out_connect[1, :, ::-1, :])
        out_connect_full.append(out_connect[2, :, :, ::-1])
        out_connect_full.append(out_connect[3, :, ::-1, ::-1])
        out_connect_full = np.asarray(out_connect_full).mean(axis=0)[np.newaxis, :, :, :]
        pred_connect = np.sum(out_connect_full, axis=1)
        pred_connect[pred_connect < 0.9] = 0
        pred_connect[pred_connect >= 0.9] = 1

        out_connect_d1_full = []
        out_connect_d1 = out_connect_d1.data.cpu().numpy()
        out_connect_d1_full.append(out_connect_d1[0, ...])
        out_connect_d1_full.append(out_connect_d1[1, :, ::-1, :])
        out_connect_d1_full.append(out_connect_d1[2, :, :, ::-1])
        out_connect_d1_full.append(out_connect_d1[3, :, ::-1, ::-1])
        out_connect_d1_full = np.asarray(out_connect_d1_full).mean(axis=0)[np.newaxis, :, :, :]
        pred_connect_d1 = np.sum(out_connect_d1_full, axis=1)
        pred_connect_d1[pred_connect_d1 < 2.0] = 0
        pred_connect_d1[pred_connect_d1 >= 2.0] = 1

        pred_full = []
        pred = output.data.cpu().numpy()
        target_n = target.cpu().numpy()
        pred_full.append(pred[0, ...])
        pred_full.append(pred[1, :, ::-1, :])
        pred_full.append(pred[2, :, :, ::-1])
        pred_full.append(pred[3, :, ::-1, ::-1])
        pred_full = np.asarray(pred_full).mean(axis=0)

        pred_full[pred_full > 0.1] = 1
        pred_full[pred_full < 0.1] = 0

        su = pred_full + pred_connect + pred_connect_d1
        su[su > 0] = 1
        
        # Convert target and su arrays to grayscale
        if target_n.shape[-1] == 3:
            target_n = target_n[..., 0]        
        # su_grayscale = su[:, :, 0]  # Extract one channel as grayscale


        print("Shape of target_n:", target_n.shape)
        print("Shape of su:", su.shape)


        evaluator.add_batch(target_n, su.astype(int))

        out_image = make_grid(image[0,:].clone().cpu().data, 3, normalize=True)
        # Call decode_segmap with the grayscale target and su arrays
        out_GT = make_grid(decode_seg_map_sequence(target_n,
                                                dataset=args.dataset), 3, normalize=False, range=(0, 255))
        out_pred_label_sum = make_grid(decode_seg_map_sequence(su,
                                                       dataset=args.dataset), 3, normalize=False, range=(0, 255))

        save_image(out_image, os.path.join(out_path, img_name + '_sat.png'))
        save_image(out_GT, os.path.join(out_path, img_name + '_GT' + '.png'))
        save_image(out_pred_label_sum, os.path.join(out_path, img_name + '_pred' + '.png'))
        # processed_image_count += image.data.shape[0]  # Increment the counter by the batch size




        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        IoU = evaluator.Intersection_over_Union()
        Precision = evaluator.Pixel_Precision()
        Recall = evaluator.Pixel_Recall()
        F1 = evaluator.Pixel_F1()

        # Update the best accuracy values
        if mIoU > best_mIoU:
            best_acc = Acc
            best_acc_class = Acc_class
            best_mIoU = mIoU
            best_IoU = IoU
            best_precision = Precision
            best_recall = Recall
            best_f1 = F1

        print('Validation:')
        # print('[numImages: %5d]' % (i * args.batch_size + image.data.shape[0]))
        print('[numImages: %5d]' % ((i + 1) * args.batch_size))  # Print the processed image count
        # print(image.data.shape[0])
        # print(i)
        # print(args.batch_size)
        # print('[numImages: %5d]' % processed_image_count)  # Print the processed image count
        print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
            .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))
        
         # Create a dictionary to store all metrics for the current frame
        frame_metrics_dict = {
            "Frame Number": frame_number,
            "Acc": Acc,
            "Acc_class": Acc_class,
            "mIoU": mIoU,
            "IoU": IoU,
            "Precision": Precision,
            "Recall": Recall,
            "F1": F1
        }

        # Append frame metrics to the list
        frame_metrics.append(frame_metrics_dict)

    # Find the frame with the highest mIoU
    best_frame_mIoU = max(frame_metrics, key=lambda x: x["mIoU"])
    best_frame_number = best_frame_mIoU["Frame Number"]
    best_frame_metrics = best_frame_mIoU

    # Create a text file to save accuracy results
    with open(os.path.join(out_path, 'frame_metrics.txt'), 'w') as f:
        f.write("Frame Metrics Results:\n")
        for frame_metrics_dict in frame_metrics:
            f.write("Frame Number: {}\n".format(frame_metrics_dict["Frame Number"]))
            f.write("Acc: {}\n".format(frame_metrics_dict["Acc"]))
            f.write("Acc_class: {}\n".format(frame_metrics_dict["Acc_class"]))
            f.write("mIoU: {}\n".format(frame_metrics_dict["mIoU"]))
            f.write("IoU: {}\n".format(frame_metrics_dict["IoU"]))
            f.write("Precision: {}\n".format(frame_metrics_dict["Precision"]))
            f.write("Recall: {}\n".format(frame_metrics_dict["Recall"]))
            f.write("F1: {}\n".format(frame_metrics_dict["F1"]))
            f.write("\n")

        f.write("Best Frame Number: {}\n".format(best_frame_number))
        f.write("Best Frame Metrics:\n")
        f.write("Acc: {}\n".format(best_frame_metrics["Acc"]))
        f.write("Acc_class: {}\n".format(best_frame_metrics["Acc_class"]))
        f.write("mIoU: {}\n".format(best_frame_metrics["mIoU"]))
        f.write("IoU: {}\n".format(best_frame_metrics["IoU"]))
        f.write("Precision: {}\n".format(best_frame_metrics["Precision"]))
        f.write("Recall: {}\n".format(best_frame_metrics["Recall"]))
        f.write("F1: {}\n".format(best_frame_metrics["F1"]))

    print("Best Accuracy:")
    print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
          .format(best_acc, best_acc_class, best_mIoU, best_IoU, best_precision, best_recall, best_f1))

    # # Save the best accuracy values to a .txt file
    # with open(os.path.join(out_path, 'Best_accuracy.txt'), 'w') as f:
    #     f.write("Best Accuracy:\n")
    #     f.write("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}\n"
    #             .format(best_acc, best_acc_class, best_mIoU, best_IoU, best_precision, best_recall, best_f1))

if __name__ == "__main__":
   main()
