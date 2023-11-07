from dataloaders.datasets import tia,tia_crop
# from dataloaders.datasets import spacenet, spacenet_crop, deepglobe, deepglobe_crop, tia
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from mypath import Path
import torchvision.transforms as transforms
import numpy as np

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def make_data_loader(args, **kwargs):

    if args.dataset == '7b52009b64fd0a2a49e6d8a939753077792b0554':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == '0ade7c2cf97f75d009975f4d720d1fa6c19f4897':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == '1b6453892473a467d07372d45eb05abc2031647a':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == '17ba0791499db908433b80f37c5fbc89b870084b':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == '356a192b7913b04c54574d18c28d46e6395428ab':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == '902ba3cda1883801594b6e1b452790cc53948fda':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == 'b1d5781111d84f7b3fe45a0852e59758cd7a87e5':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'bd307a3ec329e10a2cff8fb87480823da114f8f4':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == 'c1dfd96eea8cc2b62785275bca38ac261256e278':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == 'fa35e192121eabf3dabf9f5ea6abdbcbc107ac3b':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == 'mergedataset':
        train_set = tia_crop.Segmentation(args, split='train')
        val_set = tia_crop.Segmentation(args, split='val')
        test_set = tia.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

        
    

    else:
        raise NotImplementedError


