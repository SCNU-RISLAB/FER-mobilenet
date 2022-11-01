
import math
import numpy as np
import torchvision.models as models
import pandas as pd
import torch.utils.data as data
from torchvision import transforms, datasets
import cv2
import os
import torch
import torch.nn as nn
import image_utils
import argparse
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
# from model_v3 import mobilenet_v3_small
from Model_V3 import mobilenet_v3_large
# from Origin_MobileNetV3 import mobilenet_v3_large
from torch.utils.tensorboard import SummaryWriter
import time
from ptflops import get_model_complexity_info
from torch.hub import load_state_dict_from_url  # noqa: 401


def parse_args():  # 解析参数定义

    parser = argparse.ArgumentParser()

    parser.add_argument('--fer2013_plus_path', type=str,
                        default="/home/lenovo/LJF/Face_emotion/Resnet2MobilenetV3/datasets/fer2013_plus/",
                        help='fer2013_plus dataset path.')
    parser.add_argument('--raf_path', type=str,
                        default="/home/lenovo/LJF/Face_emotion/Resnet2MobilenetV3/datasets/raf-basic/",
                        help='Raf-DB dataset path.')

    parser.add_argument('--train_log_path', type=str,
                        default="/home/lenovo/LJF/Face_emotion/Resnet2MobilenetV3/new_newexp/train_log/file/",
                        help='train_log path.')
    parser.add_argument('--train_log_txt_path', type=str,
                        default="/home/lenovo/LJF/Face_emotion/Resnet2MobilenetV3/new_newexp/train_log/txt_file/",
                        help='save train log in txt formal')
    parser.add_argument('--confusion_matrix_path', type=str,
                        default="/home/lenovo/LJF/Face_emotion/Resnet2MobilenetV3/new_newexp/train_log/pic_file/")
    parser.add_argument('--best_model_path', type=str,
                        default="/home/lenovo/LJF/Face_emotion/Resnet2MobilenetV3/new_newexp/best_model/",
                        help='best_model_path.')

    parser.add_argument('--Model', type=str, default="Mobilenetv3", help='choose models.')
    parser.add_argument('--dataset', type=str, default="fer2013_plus",
                        help='choose dataset (fer2013_plus or raf-basic)')
    parser.add_argument("--centerloss",type=float,default=0.4, help="adjust the centerloss in loss")


    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--drop_rate', type=float, default=0.4, help='Drop out rate.')

    parser.add_argument('--num_classes', type=int, default=8, help='  fer2013_plus: 8  raf-basic: 7 ')

    return parser.parse_args()


# Fer2013_plus DataSet
class FerPlusDataSet(data.Dataset):  # 数据读取
    def __init__(self, fer_path, phase, transform=None, basic_aug=False):
        self.phase = phase  # 判断读入的数据是训练还是验证
        self.transform = transform
        self.fer_path = fer_path

        self.LabelName = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sadness', 'Anger', 'Neutral', 'Contempt']
        # self.LabelName = ['Surprise', 'Fear', 'Disgust', 'Sadness', 'Anger', 'Neutral', 'Contempt', 'Happy']
        self.file_paths = []
        self.label = []

        if self.phase == 'train':
            data_path = self.fer_path + 'Training'
        else:
            data_path = self.fer_path + 'PrivateTest'

        for files in os.listdir(data_path):
            for file in os.listdir(os.path.join(data_path, files)):
                file = os.path.join(data_path, files, file)
                labelindex = self.LabelName.index(files)
                self.file_paths.append(file)
                self.label.append(np.int64(labelindex))
        self.label = np.array(self.label)  # 参考train.py文件，此处self.label必须为numpy数组，这样后面的relabel操作才能根据索引改变标签数值，故需要进行类型转换
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.rotate, image_utils.shift, image_utils.crop,
                         image_utils.lighting_adjust, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)  # image.shape is (48, 48, 3)

        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 5)  # 随机产生一个int型数据，0或1
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)  # 图片变换

        return image, label, idx

class RafDataSet(data.Dataset):  # 数据读取
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        self.phase = phase  # 判断读入的数据是训练还是验证
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0  # 标签txt文件中 图片名所在的列数为0，即第一列
        LABEL_COLUMN = 1  # 标签txt文件中 标签值所在的列数为1，即第二列

        # read_csv：第一个参数是文件路径，sep参数设置分隔符，header=None，即指认为原始文件数据没有列索引，这样read_csv为其自动加上列索引{从0开始}
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)

        if phase == 'train':
            dataset = df[
                df[NAME_COLUMN].str.startswith('train')]  # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        # 取txt文件的第一列数据
        file_names = dataset.iloc[:, NAME_COLUMN].values
        # 取txt文件的第二列数据组成标签值列表
        self.label = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]  # split() 通过指定分隔符对字符串进行切片; split切片原因：标签txt文件注明的图片名跟数据图片jpg文件名不一致，需要切开再拼接成正确的图片名
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)  # 指向每张图片的路径，for循环不断添加图片至列表
            self.file_paths.append(path)  # 该列表存放每张图片的路径全名

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.rotate, image_utils.shift, image_utils.crop,
                         image_utils.lighting_adjust, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug  and random.uniform(0, 1) > 0.55 :
                index = random.randint(0, 5)  # 随机产生一个int型数据，0或1
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)  # 图片变换

        return image, label, idx

def run_training():
    # 环境配置
    args = parse_args()
    if args.num_classes  == 7:
       args.dataset="raf-basic"
    else:
       args.dataset = "fer2013_plus"

    if args.dataset == "fer2013_plus":
        dataset_test="FERPlus"
    else:
        dataset_test="RAF-DB"



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = mobilenet_v3_large(num_classes=8)

    FLOPs, params_1 = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)

    print("网络模型参数指标 : GFLOPs : %s    params : %s" % ( FLOPs, params_1))


    data_transforms = transforms.Compose([  # 图像预处理transforms，用Compose整合图像处理多个步骤
        transforms.ToPILImage(),  # convert a tensor to PIL image
        transforms.Resize((224, 224)),  # image scale resize to 224 * 224
        transforms.ToTensor(),  # convert a PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))])


    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # 加载不同的训练数据集

    if args.dataset == "fer2013_plus":
        val_dataset = FerPlusDataSet(args.fer2013_plus_path, phase='test', transform=data_transforms_val,
                                     basic_aug=False)

    elif args.dataset == "raf-basic":
        val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val, basic_aug=False)


    # print('Validation set size:', val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    model_save_path = "/home/lenovo/LJF/Face_emotion/Resnet2MobilenetV3/new_newexp/best_model/10_30 FER2013 0.6 SC/epoch36_best_acc 0.8771.pth"


    checkpoint = torch.load(model_save_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(device)


    if args.dataset == "fer2013_plus":
        labels_name = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sadness', 'Anger', 'Neutral', 'Contempt']
        # labels_name = ['Surprise', 'Fear', 'Disgust', 'Sadness', 'Anger', 'Neutral', 'Contempt','Happy']

    elif args.dataset == "raf-basic":
        labels_name = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']



    targets_test = []
    predict_test = []
    best_acc = 0

        # Start Validation
    with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            net.eval()
            cost_time_list=[]
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                # _, outputs = net(imgs.to(device))
                imgs = imgs.to(device)
                targets = targets.to(device)
                torch.cuda.synchronize()
                time_start = time.time()
                feature_val,outputs = net(imgs)

                _, predicts = torch.max(outputs, 1)

                torch.cuda.synchronize()
                time_end = time.time()
                time_sum = time_end - time_start
                time_sum=time_sum/64
                cost_time_list.append(time_sum)

                iter_cnt += 1
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)
                targets_test.extend(targets.data.cpu().numpy())
                predict_test.extend(predicts.data.cpu().numpy())

            cost_time=0
            for i in range(5,len(cost_time_list)):
                cost_time += cost_time_list[i]

            cost_time=cost_time/(len(cost_time_list)-5)
            print("模型推理速度 : {:.5f}秒/张图片".format(cost_time))

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            print("测试集准确率:",acc)
            matrix = confusion_matrix(np.array(targets_test).tolist(), np.array(predict_test).tolist())
            plot_confusion_matrix(matrix, labels_name,  " Confusion Matrix")  # 对准确率最高的验证结果绘制混淆矩阵
            # plt.show()

def plot_confusion_matrix(cm, classes, title=None, cmap=plt.cm.YlGnBu):  # 混淆矩阵的绘制
    plt.rc('font', family='Times New Roman', size='12')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                  cm[i, j] = 0



    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink=0.5)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
                if int(cm[i, j] * 100 + 0.5) > 0:
                    ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")


    fig.tight_layout()



if __name__ == "__main__":
    run_training()
    # tensorboard  --logdir="./train_log"