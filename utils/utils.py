from data_loader.imagedata import ImageData
from data_loader.fer_imagedata import FERimageData
from data_loader.raf_imagedata import RafDataset
from torch.utils.data import DataLoader

import path_datasets as path
def dataset_loader(dataset_name,batch_size,train_transformation,test_transformation):
    classes = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', 'NE']
    if dataset_name == 'ck':
        train_dataset = ImageData(csv_file=path.path_ck_train_csv, img_dir=path.path_ck_train_image,
                                  datatype='ck_train', transform=train_transformation)
        test_dataset = ImageData(csv_file=path.path_ck_test_csv, img_dir=path.path_ck_test_image, datatype='ck_val',
                                 transform=test_transformation)
        classes = ['AN', 'DI', 'FE', 'HA', 'SA', 'SU', 'NE']

    elif dataset_name == 'fer':
        train_dataset = FERimageData(csv_file=path.path_fer_train_csv, img_dir=path.path_fer_train_image,
                                     datatype='train', transform=train_transformation)
        test_dataset = FERimageData(csv_file=path.path_fer_test_csv, img_dir=path.path_fer_test_image, datatype='val_1',
                                    transform=test_transformation)

    elif dataset_name == 'raf':
        train_dataset = RafDataset(path=path.path_raf, phase='train', transform=test_transformation)
        test_dataset = RafDataset(path=path.path_raf, phase='test', transform=test_transformation)


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    return train_dataset,test_dataset,train_loader, test_loader,classes