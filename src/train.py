
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
    parser.add_argument('--num_classes', type=int, default=8, help='fer2013_plus: 8  raf-basic: 7 ')
    parser.add_argument("--centerloss",type=float,default=0.4, help="adjust the centerloss in loss")
    parser.add_argument('--improve_operation', type=str,
                        default=" train result",
                        help='the improved operation')

    parser.add_argument('--checkpoint', type=str, default="checkpoint", help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str,
                        default="models/pretrained weight/mobilenet_v3_large-8738ca79.pth",
                        help='Pretrained weights')


    parser.add_argument('--beta', type=float, default=0.7,
                        help='Ratio of high importance group in one mini-batch.')  # 论文里的β，高权重组别的比例
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--patience', default=10, type=int, help='patience for val_acc not improve')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0.4, help='Drop out rate.')
    parser.add_argument('--save_gate', type=float, default=0.85, help='only over this value could be saved.')

    parser.add_argument('--margin_1', type=float, default=0.07,
                        help='Rank regularization margin. Details described in the paper.')  # 论文里的δ1
    parser.add_argument('--margin_2', type=float, default=0.2,
                        help='Relabeling margin. Details described in the paper.')  # 论文里的δ2
    parser.add_argument('--relabel_epoch', type=int, default=10,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    return parser.parse_args()


# Fer2013_plus DataSet
class FerPlusDataSet(data.Dataset):  # 数据读取
    def __init__(self, fer_path, phase, transform=None, basic_aug=False):
        self.phase = phase  # 判断读入的数据是训练还是验证
        self.transform = transform
        self.fer_path = fer_path

        self.LabelName = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sadness', 'Anger', 'Neutral', 'Contempt']
        self.file_paths = []
        self.label = []

        if self.phase == 'train':
            data_path = self.fer_path + 'Training'
        else:
            data_path = self.fer_path + 'PublicTest'

        for files in os.listdir(data_path):
            for file in os.listdir(os.path.join(data_path, files)):
                file = os.path.join(data_path, files, file)
                labelindex = self.LabelName.index(files)
                self.file_paths.append(file)
                self.label.append(np.int64(labelindex))
        self.label = np.array(self.label)  # 参考train.py文件，此处self.label必须为numpy数组，这样后面的relabel操作才能根据索引改变标签数值，故需要进行类型转换
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.rotate, image_utils.shift,image_utils.crop,

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    args = parse_args()
    net = mobilenet_v3_large(num_classes=args.num_classes)
    # net = mobilenet_v3_large(num_classes=args.num_classes)
    # net=models.inception_v3(True,True)
    # 统计模型参数量的大小  flops 和 params
    FLOPs, params_1 = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print("%s flops : %s    params : %s" % (args.improve_operation, FLOPs, params_1))

    # MobileNetV3 pretrained

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        net_dict = net.state_dict()
        pre_weights = torch.load(args.pretrained, map_location=device)

        pre_weights = {k: v for k, v in pre_weights.items() if
                        ("features.14" not in k) and ("features.13" not in k) and (("features.16" not in k))  and ("features.15" not in k)

                       }

        print(pre_weights.keys())
        net_dict.update(pre_weights)
        net.load_state_dict(pre_weights, False)

    # 训练集数据预处理、读取；12271张训练集图片

    data_transforms = transforms.Compose([  # 图像预处理transforms，用Compose整合图像处理多个步骤
        transforms.ToPILImage(),  # convert a tensor to PIL image
        transforms.Resize((224, 224)),  # image scale resize to 224 * 224
        transforms.ToTensor(),  # convert a PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))])

    # 加载不同的训练数据集
    if args.dataset == "fer2013_plus":
        train_dataset = FerPlusDataSet(args.fer2013_plus_path, phase='train', transform=data_transforms, basic_aug=True)

    elif args.dataset == "raf-basic":
        train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   num_workers=args.workers,
                                   shuffle=True,
                                   pin_memory=True)

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

    print('Validation set size:', val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    params = net.parameters()
    # 优化器选择
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.patience,
                                                           verbose=True, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)  # 自适应调整学习率
    net = net.to(device)

    criterion = nn.NLLLoss()


    # 数据可视化需要的参数变量定义
    if args.dataset == "fer2013_plus":
        labels_name = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sadness', 'Anger', 'Neutral', 'Contempt']

    elif args.dataset == "raf-basic":
        labels_name = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    targets_test = []
    predict_test = []
    best_acc = 0


    new_path = args.train_log_path + args.improve_operation
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    # creat models path
    model_path = args.best_model_path + args.improve_operation
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # tensorboard 训练过程可视化
    writer = SummaryWriter(new_path)
    start_epoch = 1

    for i in range(start_epoch, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        net.train()  # 开启训练模式****************************************88此处调用网络框架
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0)  # 一个epoch下的batch_size大小
            iter_cnt += 1  # 一个epoch下的iter数
            optimizer.zero_grad()
            imgs = imgs.to(device)

            feature_train, outputs = net(imgs)

            targets = targets.to(device)

            crossentropy_loss = criterion(outputs, targets)

            y = targets.float()

            loss_center = center_loss(feature_train, y, args.centerloss)  # 比重2可以给小一些，比如0.5

            loss = crossentropy_loss + loss_center  # CELoss(相当于softmax_loss) + Center loss

            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            running_loss += loss
            _, predicts = torch.max(outputs, 1)  # 返回输入tensor中所有元素的最大值
            correct_num = torch.eq(predicts, targets).sum()  # 两个张量逐个比较，对应元素是相同的就返回True 否则返回False
            correct_sum += correct_num  # 统计一个batch_size下的图片表情预测正确个数

        # 一个epoch训练完后，计算准确率，损失值，学习率更新
        # scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        writer.add_scalar("train_acc", acc, i)
        writer.add_scalar("train_loss", running_loss, i)
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
        print("epoch:", i, "i:", iter_cnt, "total_loss:", loss.item(),
              "Softmax_loss", crossentropy_loss.item(), "center_loss", loss_center.item())

        # Start Validation
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            net.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                # _, outputs = net(imgs.to(device))

                feature_val, outputs = net(imgs.to(device))

                targets = targets.to(device)

                val_crossentropy_loss = criterion(outputs, targets)
                y_f = targets.float()
                loss_center_2 = center_loss(feature_val, y_f, 0.5)

                loss = val_crossentropy_loss + loss_center_2

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)
                targets_test.extend(targets.data.cpu().numpy())
                predict_test.extend(predicts.data.cpu().numpy())

            val_loss = val_loss / iter_cnt
            scheduler.step(val_loss)  # 当val_loss 10个epoch 没有下降时 就调整学习率为原来的0.1倍
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            writer.add_scalar("val_acc", acc, i)
            writer.add_scalar("val_loss", val_loss, i)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, val_loss))

            # save models

            if acc > args.save_gate and acc > best_acc:
                best_acc = acc
                torch.save({'iter': i,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join(model_path, "epoch" + str(i)  + "_best_acc " + str(
                               best_acc) + ".pth"))
                print('Model saved.')

    print("the best val_acc :{}".format(best_acc))

    # save train log
    train_log_txt_path = model_path+"//" + args.improve_operation + str(best_acc) + ".txt"
    train_log_file = open(train_log_txt_path, "w")
    train_log_file.write(
        "case : {} \n  best acc : {}\n   FLOPs : {}\n   params : {}\n  batch_size{}\n".format(args.improve_operation,
                                                                                              best_acc, FLOPs,
                                                                                              params_1, args.batch_size)
    )
    train_log_file.close()
    writer.close()


def center_loss(feature, label, lambdas):
    center = nn.Parameter(torch.randn(int(max(label).item() + 1), feature.shape[1]), requires_grad=True)  # .cuda()

    center = center.to("cuda")

    center_exp = center.index_select(dim=0, index=label.long())

    count = torch.histc(label, bins=int(max(label).item() + 1), min=0, max=int(max(label).item()))

    count_exp = count.index_select(dim=0, index=label.long())

    loss = lambdas / 2 * torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp))
    return loss


if __name__ == "__main__":
    matrix = confusion_matrix([0, 2, 3, 1, 5, 1, 2], [1, 1, 1, 1, 1, 1, 1])
    run_training()
    # tensorboard  --logdir="./train_log"