import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.coanet import *
from utils.loss import SegmentationLosses, dice_bce_loss
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torch.utils.data import Dataset
import math
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import DataParallel
import random
from torch.utils.data.sampler import Sampler


# import pdb

# latest with update how the num of images being trained

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
            self.batch_size = 4
            self.cuda = not self.no_cuda and torch.cuda.is_available()
            # self.gpu_ids = []  # Use an empty list to indicate using all available GPUs
            # self.gpu_ids = ''  # Use an empty list to indicate not using GPUs
            self.gpu_ids = [0]  # Use an empty list to indicate using one  GPUs    
            self.dataset = 'mergedataset'
            self.base_size = 280
            self.crop_size = 280
            # self.base_size = 512
            # self.crop_size = 512
            self.sync_bn = False
            self.freeze_bn = False
            self.backbone = 'resnet'
            self.out_stride = 8
            self.loss_type = 'con_ce'
            self.epochs = 30
            self.start_epoch = 0
            self.use_balanced_weights = False
            self.lr = 0.01
            self.lr_scheduler = 'poly'
            self.momentum = 0.9
            self.weight_decay = 5e-4
            self.nesterov = False
            self.seed = 1
            self.resume = None
            self.checkname = None
            self.ft = False
            self.eval_interval = 1
            self.no_val = False
            self.workers = 16


# class SegmentationSubset(Dataset):
#     def __init__(self, dataset, start_idx, end_idx):
#         self.dataset = dataset
#         self.start_idx = start_idx
#         self.end_idx = end_idx

#     def __len__(self):
#         return self.end_idx - self.start_idx

#     def __getitem__(self, index):
#         return self.dataset[self.start_idx + index]

# class SubsetDataset(Dataset):
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = indices

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, index):
#         dataset_index = self.indices[index]
#         return self.dataset[dataset_index]



class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # Determine the total number of images in the training dataset
        self.total_images = len(self.train_loader.dataset)

        # Define the maximum size of each subset
        self.max_subset_size = 300  # Add this line to define max_subset_size

        # Calculate the number of subsets needed
        self.num_subsets = self.total_images // self.max_subset_size
        self.remaining_images = self.total_images % self.max_subset_size

        # print("remaining images: ",self.remaining_images)

        self.subset_loaders = []  # To store all the subset loaders
        self.current_subset_index = 0  # Add a variable to keep track of the current subset index
        self.total_images_trained = 0


        # Create subsets and loaders for each subset
        for i in range(self.num_subsets):
            start_idx = i * self.max_subset_size
            end_idx = (i + 1) * self.max_subset_size

            subset = Subset(self.train_loader.dataset, list(range(start_idx, end_idx)))
            subset_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

            self.subset_loaders.append(subset_loader)

            print(f"Subset {i+1}: Number of images = {len(subset_loader.dataset)}")


        # # Create a loader for the last subset with the remaining images
        # if self.remaining_images > 0:
        #     start_idx = self.num_subsets * self.max_subset_size
        #     end_idx = self.num_subsets * self.max_subset_size + self.remaining_images

        #     last_subset = Subset(self.train_loader.dataset, list(range(start_idx, end_idx)))
        #     last_subset_loader = DataLoader(last_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

        #     self.subset_loaders.append(last_subset_loader)
            
        #     print(f"Number of images = {len(subset_loader.dataset)}")

        # Create a loader for the last subset with the remaining images
        if self.remaining_images > 0:
            start_idx = self.num_subsets * self.max_subset_size
            end_idx = self.num_subsets * self.max_subset_size + self.remaining_images

            last_subset = Subset(self.train_loader.dataset, list(range(start_idx, end_idx)))
            last_subset_loader = DataLoader(last_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

            self.subset_loaders.append(last_subset_loader)
            
            print(f"Number of images in last subset = {len(last_subset_loader.dataset)}")  # Print the number of images in the last subset

        # Print remaining_images, start_idx, and end_idx to check their values
        # print(f"remaining_images = {self.remaining_images}")
        # print(f"start_idx = {start_idx}")
        # print(f"end_idx = {end_idx}")



        # Define network
        model = CoANet(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_2x_lr_params(), 'lr': args.lr * 2}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            # classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.py')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = dice_bce_loss()
        self.criterion_con = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(2)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda and args.gpu_ids:  # Check if gpu_ids is not empty
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()  # Run on CPU

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
                # self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        # tbar = tqdm(self.train_loader)
        # num_img_tr = len(self.train_loader)
        for subset_index, subset_loader in enumerate(self.subset_loaders):
            # tbar = tqdm(subset_loader)  # Create tqdm progress bar for the current subset
            num_img_tr = len(subset_loader)

            self.current_subset_index += 1

            # Shuffle the train_loader
            # tbar = list(subset_loader)
            # random.shuffle(tbar)
            # tbar = tqdm(subset_loader, desc='\r')

            # Create a custom data sampler to shuffle the test_loader
            subset_sampler = RandomSamplerWithReplacement(subset_loader.dataset)

            # Create DataLoader with the custom data sampler
            shuffled_train_loader = DataLoader(dataset=subset_loader.dataset, batch_size= self.args.batch_size, sampler=subset_sampler, num_workers=0, pin_memory=False)

            # Create a tqdm object for the shuffled test_loader with a progress bar
            tbar = tqdm(shuffled_train_loader, desc='\r')

            for i, sample in enumerate(tbar):
                image, target, con0, con1, con2, con_d1_0, con_d1_1, con_d1_2 = \
                    sample['image'], sample['label'], sample['connect0'], sample['connect1'], sample['connect2'],\
                    sample['connect_d1_0'], sample['connect_d1_1'], sample['connect_d1_2']

                subset_index = i // self.max_subset_size  # Determine which subset this batch belongs to
                global_batch_index = subset_index * self.max_subset_size + i
                num_images_in_subset = len(subset_loader.dataset)
                # self.total_images_trained += num_images_in_subset
                self.total_images_trained += self.args.batch_size  # Increment by the batch size


                        
                connect_label = torch.cat((con0, con1, con2), 1)
                connect_d1_label = torch.cat((con_d1_0, con_d1_1, con_d1_2), 1)
                if self.args.cuda:
                    image, target, connect_label, connect_d1_label = image.cuda(), target.cuda(), connect_label.cuda(), connect_d1_label.cuda()
                

                # print(f"Processing batch {global_batch_index} in Subset {subset_index + 1}/{len(subset_loader)}, Total images in subset: {num_images_in_subset}")
                print(f"Processing batch {global_batch_index} in Subset {self.current_subset_index}/{len(self.subset_loaders)}, Total images in subset: {num_images_in_subset}")
                # pdb.set_trace()

                # print("Processing batch:", i)
                # print("Image shape:", image.shape)
                # print("Target shape:", target.shape)
                # print("Connect label shape:", connect_label.shape)
                # print("Connect_d1 label shape:", connect_d1_label.shape)
                
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                self.optimizer.zero_grad()
                output, out_connect, out_connect_d1 = self.model(image)
                target = torch.unsqueeze(target, 1)
                if target.shape[-1] == 3: 
                    target = target[..., 0] 
                loss1 = self.criterion(output, target)
                loss2 = self.criterion_con(out_connect, connect_label)
                loss3 = self.criterion_con(out_connect_d1, connect_d1_label)
                lad = 0.2
                loss = loss1 + lad*(0.6*loss2 + 0.4*loss3)
                loss.backward()
                self.optimizer.step()
                train_loss1 += loss1.item()
                train_loss2 += lad * 0.6 * loss2.item()
                train_loss3 += lad * 0.4 * loss3.item()
                train_loss += loss.item()
                tbar.set_description('Train loss: %.3f, loss1: %.6f, loss2: %.3f, loss3: %.3f' %
                                    (train_loss / (i + 1), train_loss1 / (i + 1), train_loss2 / (i + 1), train_loss3 / (i + 1)))
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

                pred = output.data.cpu().numpy()
                target_n = target.cpu().numpy()
                # Add batch sample into evaluator
                pred[pred > 0.1]=1
                pred[pred < 0.1] = 0
                self.evaluator.add_batch(target_n, pred)

            # torch.cuda.empty_cache() #uncomment it if using GPU


        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/loss1_epoch', train_loss1, epoch)
        self.writer.add_scalar('train/loss2_epoch', train_loss2, epoch)
        self.writer.add_scalar('train/loss3_epoch', train_loss3, epoch)
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        self.writer.add_scalar('train/Acc', Acc, epoch)
        self.writer.add_scalar('train/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('train/IoU', IoU, epoch)
        self.writer.add_scalar('train/Precision', Precision, epoch)
        self.writer.add_scalar('train/Recall', Recall, epoch)
        self.writer.add_scalar('train/F1', F1, epoch)
        print('Train:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, self.total_images_trained))
        print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))
        print('Loss: %.3f, Loss1: %.6f, Loss2: %.3f, Loss3: %.3f' % (train_loss, train_loss1, train_loss2, train_loss2))

        if self.args.no_val:
            # save checkpoint every epoch
            self.model.to('cpu')  # Move the model to CPU
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        # tbar = tqdm(self.val_loader, desc='\r')
        test_loss1 = 0.0
        test_loss2 = 0.0
        test_loss3 = 0.0
        test_loss = 0.0
        num_img_tr = len(self.val_loader)

        # Shuffle the train_loader
        # tbar = list(self.val_loader)
        # random.shuffle(tbar)          
        # tbar = tqdm(self.val_loader, desc='\r')


        # Create a custom data sampler to shuffle the test_loader
        subset_sampler = RandomSamplerWithReplacement(self.val_loader.dataset)

        # Create DataLoader with the custom data sampler
        shuffled_val_loader = DataLoader(dataset=self.val_loader.dataset, batch_size=self.args.batch_size, sampler=subset_sampler, num_workers=0, pin_memory=False)

        # Create a tqdm object for the shuffled test_loader with a progress bar
        tbar = tqdm(shuffled_val_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target, con0, con1, con2, con_d1_0, con_d1_1, con_d1_2 = \
                sample[0]['image'], sample[0]['label'], sample[0]['connect0'], sample[0]['connect1'], sample[0]['connect2'], \
                sample[0]['connect_d1_0'], sample[0]['connect_d1_1'], sample[0]['connect_d1_2']
            connect_label = torch.cat((con0, con1, con2), 1)
            connect_d1_label = torch.cat((con_d1_0, con_d1_1, con_d1_2), 1)

            if self.args.cuda:
                image, target, connect_label, connect_d1_label = image.cuda(), target.cuda(), connect_label.cuda(), connect_d1_label.cuda()
            with torch.no_grad():
                output, out_connect, out_connect_d1 = self.model(image)
            target = torch.unsqueeze(target, 1)
            if target.shape[-1] == 3: 
                target = target[..., 0] 
            loss1 = self.criterion(output, target)
            loss2 = self.criterion_con(out_connect, connect_label)
            loss3 = self.criterion_con(out_connect_d1, connect_d1_label)
            lad = 0.2
            loss = loss1 + lad * (0.6*loss2 + 0.4*loss3)
            test_loss1 += loss1.item()
            test_loss2 += lad * 0.6 * loss2.item()
            test_loss3 += lad * 0.4 * loss3.item()
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f, loss1: %.6f, loss2: %.3f, loss3: %.3f' % (test_loss / (i + 1), test_loss1 / (i + 1), test_loss2 / (i + 1), test_loss3 / (i + 1)))
            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            # Add batch sample into evaluator
            pred[pred > 0.1]=1
            pred[pred < 0.1] = 0
            self.evaluator.add_batch(target_n, pred)

            if i % (num_img_tr // 1) == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i,
                                             split='val')

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/loss1_epoch', test_loss1, epoch)
        self.writer.add_scalar('val/loss2_epoch', test_loss2, epoch)
        self.writer.add_scalar('val/loss3_epoch', test_loss3, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/IoU', IoU, epoch)
        self.writer.add_scalar('val/Precision', Precision, epoch)
        self.writer.add_scalar('val/Recall', Recall, epoch)
        self.writer.add_scalar('val/F1', F1, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))
        print('Loss: %.3f, Loss1: %.3f, Loss2: %.3f, Loss3: %.3f' % (test_loss, test_loss1, test_loss2, test_loss3))

        new_pred = IoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
    

def get_available_gpu_indices():
    num_gpus = torch.cuda.device_count()
    available_gpu_indices = []

    for gpu_index in range(num_gpus):
        available_gpu_indices.append(gpu_index)

    return available_gpu_indices

def main():
    args = Args()

    # Get available GPU indices
    # available_gpus = get_available_gpu_indices()
    # print("Available GPU indices:", available_gpus)

    # # Set up your args.gpu_ids based on available GPUs
    # args.gpu_ids = available_gpus

    # Print the number of GPUs being used   
    if args.gpu_ids:
        args.cuda = True  # Set cuda to True if using GPU
        print(f"Using {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
    else:
        args.cuda = False  # Set cuda to False if not using GPU
        print("Not using GPUs.")


    if args.checkname is None:
        args.checkname = 'CoANet-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        # torch.cuda.empty_cache()
        trainer.current_subset_index = 0
        trainer.total_images_trained = 0  # Initialize a variable to keep track of total images trained


        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


 

if __name__ == "__main__":
        main()






