import pickle
import numpy as np
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras import callbacks
import os.path
import csv
import cv2
import glob
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from matplotlib import pyplot

SEED = 13

def horizontal_flip(img, degree):
    '''
    按照50%概率水平翻转图像
    :param img: 图像
    :param degree: 输入推向的转动角度
    :return:
    '''
    choice = np.random.choice([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return img, degree


def random_brightness(img, degree):
    '''
    图像增强，调整强度于0.1~1之间
    :param img:输入图像
    :param degree:转动角度
    :return:
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 调整亮度V: alpha*V
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)
    v = hsv[:, :, 2]  # 取出hsv的第三个通道数据
    v = v * alpha
    hsv[:, :, 2] = v.astype('uint8')  # 图像每个点是整数，转换为uint8
    # 转回为rgb图像
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)

    return rgb, degree


def left_right_random_swap(img_address, degree, degree_corr=1.0/4):
    '''
    随机从左中右三幅图中选择一张，并转动相应的角度
    :param img_address: 中间图像存储的路径
    :param degree:中间图像转动角度
    :param degree_corr:转动角度调整值
    :return:
    '''
    swap = np.random.choice(['L', 'R', 'C'])

    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        corrected_label = np.arctan(math.tan(degree) + degree_corr)
        return img_address, corrected_label
    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        corrected_label = np.arctan(math.tan(degree) - degree_corr)
        return img_address, corrected_label
    else:
        return img_address, degree


def discard_zero_steering(degrees, rate):
    '''
    从角度为零的index中所及选择部分index返回
    :param degrees: 输入的角度值
    :param rate:丢弃率
    :return:
    '''
    steering_zero_idx = np.where(degrees==0)  # 选中那些degree为零的index
    steering_zero_idx = steering_zero_idx[0]
    size_del = int(len(steering_zero_idx)*rate)

    return np.random.choice(steering_zero_idx, size=size_del, replace=False)


def get_model(shape):
    '''
    预测方向盘角度：以图像作为输出，预测方向盘转动叫
    :param shape:输入图像的储存，如(128, 128, 3)3 通道
    :return model: 生成模型
    '''

    model = Sequential()

    model.add(Conv2D(8, (5, 5), strides=(1, 1), padding="valid", activation='relu', input_shape=shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (5, 5), strides=(1, 1), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (4, 4), strides=(1, 1), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='linear'))  # 只需要输出一个方向盘的角度，其角度为小数数值，故用linear和一个节点的全连接层

    # sgd = SGD(lr=0.000001)
    # model.compile(optimizer=sgd, loss="mean_squared_error")  # 拟合问题，使用均方误差层。
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error')
    return model


def image_transformation(img_address, degree, data_dir):
    '''
    读入图像
    :param img_address:图像地址
    :param label:图像标签
    :param data_dir:文件夹地址
    :return:
    '''
    img_address, degree = left_right_random_swap(img_address, degree)  # 三个视角随机来一个
    img = cv2.imread(img_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv->numpy(BGR-
    img, degree = random_brightness(img, degree)  # 调整亮度
    img, degree = horizontal_flip(img, degree)  # 水平翻转
    return img, degree


def batch_generator(x, y, batch_size, shape, training=True,
                    data_dir="data/", monitor=True,
                    yieldXY=True, discard_rate=0.95):
    '''
    产生批处理数据的generator
    :param x: 文件路径list
    :param y: 方向盘角度
    :param batch_size: 批处理大小
    :param shape: 输入图像的尺寸（长×宽×通道）
    :param training: 
                True —— 产生训练数据
                Flase —— 产生测试数据
    :param data_dir: 数据目录，包含一个IMG文件夹
    :param monitor: 是否保存一个batch的样本为'X_batch_sample.npy' 和 'y_bag.npy'
    :param yieldXY: 
                True —— 返回(X, Y)
                False —— 返回X
    :return: 
    '''

    # 训练时洗牌
    if training:
        y_bag = []
        x, y = shuffle(x, y)
        rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
        new_x = np.delete(x, rand_zero_idx, axis=0)
        new_y = np.delete(y, rand_zero_idx, axis=0)
    else:
        new_x = x
        new_y = y

    offset = 0  # 计数<=batchsize
    while True:
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))

        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]

            if training:
                img, img_steering = image_transformation(img_address, img_steering, data_dir)
            else:
                img = cv2.imread((data_dir + img_address))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 只保留中间的80*240并正规化

            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1]))/255.-0.5

            Y[example] = img_steering
            if training:
                y_bag.append(img_steering)

            '''数据全部读完，从头开始'''
            if (example + 1) + offset >= len(new_y) - 1:
                x, y = shuffle(x, y)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                offset = 0

        if yieldXY:
            yield (X, Y)
        else:
            yield X

        offset = offset + batch_size
        if training:
            np.save('y_bag.npy', np.array(y_bag))
            np.save('Xbatch_sample.npy', X)


if __name__ == '__main__':
    #  读入csv
    data_path = 'data/'
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)

    log = np.array(log)
    # 去除第一行，表格头
    log = log[1:, :]

    # 判断图像文件数量是否等于csv文件中记录的数量
    ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    print(len(ls_imgs),len(log)*3)
    assert len(ls_imgs)-3 == (len(log) * 3), "输入图像与图像数据文件不匹配"

    # 使用20%数据作为validation
    validatation_ratio = 0.2
    shape = (128, 128, 3)
    batch_size = 64
    nb_epoch = 1000

    x_ = log[:, 0]
    y_ = log[:, 3].astype(float)
    x_, y_ = shuffle(x_, y_)
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validatation_ratio, random_state=SEED)

    print('batch size:{}'.format(batch_size))
    print('Train set size:{} | Validation set size: {}'.format(len(X_train), len(X_val)))

    samples_per_epoch = batch_size
    # 使得validation数据量大小为batch_size的整数倍
    nb_val_samples = len(y_val) - len(y_val) % batch_size
    model = get_model(shape)
    print(model.summary())

    # 根据validation loss 保存最优模型
    save_best = callbacks.ModelCheckpoint('best_model.h5',
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='min')

    # 如训练持续没有validation loss提升，则题前结束训练
    early_stop = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=30,
                                         verbose=0,
                                         mode='auto')
    # callbacks_list = [early_stop, save_best]
    callbacks_list = [early_stop, save_best]

    # 使用训练数据训练
    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, shape, training=True),
                                  steps_per_epoch=samples_per_epoch,
                                  validation_steps=nb_val_samples // batch_size,
                                  validation_data=batch_generator(X_val, y_val, batch_size, shape, training=True,
                                                                  monitor=False),
                                  epochs=nb_epoch,
                                  verbose=1,
                                  callbacks=callbacks_list)

    with open('./trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title("model train VS validation loss")
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('train_val_loss.jpg')

    # 保存模型
    with open('model.json', 'w') as f:
        f.write(model.to_json())
    model.save('model.h5')
    print('Done!')