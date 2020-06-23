import numpy as np
import pylab
import matplotlib.pyplot as plt
#利用minst的数据存放特点进行读取，图片集60000个28*28字节图片，测试集10000个1字节标签
def read_image(filename, offset, amount):
    image = np.zeros((amount,28*28)) #创建amount行，28*28列的训练ndarray数组
    with open(filename, 'rb') as pf:
        pf.seek(4+4*3 + 28 * 28 * offset)#定位到第offset图片处
        for ind in range(amount):
            for row in range(28*28):
                data = pf.read(1)
                pix = int.from_bytes(data, byteorder='big')#读取大端方式存储的数据
                if pix > 50:#过滤
                    image[ind][row] = 1

            print('已读取第', ind+1, '张图片<<<  ', '剩余图片', amount-ind-1, sep='>>>')
        print('读取结束')
    return image

def read_label(filename, offset, amount):
    image = np.zeros(amount) #创建amount个数据的标签数组
    with open(filename, 'rb') as pf:
        pf.seek(4 + 4 + offset)#定位到第offset图片的标签处
        for ind in range(amount):
                data = pf.read(1)
                pix = int.from_bytes(data, byteorder='big')#读取大端方式存储的数据
                image[ind] = pix
                print('已读取第', ind+1, '张图片<<<  ', '剩余图片', amount-ind-1, sep='>>>')
        print('读取结束')
    return image

def classify(test_arr,train_arr, label_arr, K):
    sum_arr = np.sum(np.abs(test_arr - train_arr) ** 2, axis=1) ** (1/2)#axis为1，用于对行求和，即欧式求和
    find_arr = np.argsort(sum_arr)[:K]#排序后切出前k个最小的数字的下标
    label = label_arr[find_arr]
    l_sort = np.zeros(10)#存放0~9十个数的权重，用于排序
    for i in range(9):
        p = label[label == i].size / K #计算每个数在K中的权重
        l_sort[i] = p
    return np.argsort(l_sort)[9]

train_image = './train-images.idx3-ubyte'
train_lable = './train-labels.idx1-ubyte'
test_image = './t10k-images.idx3-ubyte'
test_label = './t10k-labels.idx1-ubyte'
offset, amount =map(int,input('要从第几张图片开始训练，训练图片需要多少？').split())
train_image_arr = read_image(train_image, offset, amount)
train_label_arr = read_label(train_lable, offset, amount)
offset, amount =map(int,input('要从第几张图片开始测验，测验图片需要多少？').split())

test_image_arr = read_image(train_image, offset, amount)
test_label_arr = read_label(train_lable, offset, amount)
WRONG_NUM = 0
for i in range(amount):
    label = classify(test_image_arr[i], train_image_arr , train_label_arr , 5)
    if test_label_arr[i] == label:
        print(i,'TURE')
    else:
        print(i, 'WRONG!!!!!')
        WRONG_NUM += 1
print("已成功识别结束%d张图片<<<<"%(amount),'错误率为', (float(WRONG_NUM/amount) * 100), '%', sep='')


