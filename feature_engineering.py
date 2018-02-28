import pickle
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def middle_block(data,x,y,width):
    x_max = 548
    y_max = 421
    r = int((width-1)/2)
    x_up = max(0,x-r)
    x_down = min(x_max,x+r+1)
    y_left = max(0,y-r)
    y_right = min(y_max,y+r+1)
    block = data[x_up:x_down,y_left:y_right]
    # find the index
    # index_big = np.argmax(block)
    # x_big = index_big//(y_right-y_left)+x_up
    # y_big = index_big%(y_right-y_left)+y_left
    # index_small = np.argmin(block)
    # x_small = index_small//(y_right-y_left)+x_up
    # y_small = index_small%(y_right-y_left)+y_left
    # # find the dist
    # bigpoint = np.array((x_big, y_big))
    # smallpoint = np.array((x_small, y_small))
    # thispoint = np.array((x, y))
    # dist_big = np.float16(np.linalg.norm(bigpoint - thispoint))
    # dist_small = np.float16(np.linalg.norm(thispoint - smallpoint))
    # # weather
    # current_weather = data[x][y]
    avewind = np.mean(block)
    varwind = np.std(block)
    # value_big = data[x_big][y_big]
    # value_small = data[x_small][y_small]
    # add_big = np.float16(value_big - current_weather)
    # add_small = np.float16(current_weather - value_small)

    #assert add_small>=0
    #assert add_big>=0
    return [avewind,varwind]
    #return [avewind,dist_big,dist_small,varwind]
    #return [add_big,add_small]


def windeye_search(data,x,y,width):
    x_max = 548
    y_max = 421
    r = int((width-1)/2)
    x_up = max(0,x-r)
    x_down = min(x_max,x+r+1)
    y_left = max(0,y-r)
    y_right = min(y_max,y+r+1)
    block = data[x_up:x_down,y_left:y_right]
    index = np.argmax(block)
    x_s = index//(y_right-y_left)+x_up
    y_s = index%(y_right-y_left)+y_left

    if x_s == x and y_s == y:
        return [x_s,y_s,np.max(block)-width/2,np.max(block)]
    else:

        return windeye_search(data,x_s,y_s,width)

def windeye(base_weather,sample):
    # 风眼特征
    # 1 风眼离我的距离
    # 2 风眼风速相对我的大小

    # 风速最大的地方成为凤眼
    # 参数
    # 风团最小直径，与风团的移动速度有关
    width = 9
    # 风团集合
    feature = []
    x_max = 548
    y_max = 421

    for d in range(1,6):
        for h in tqdm(range(3,21)):
            data = base_weather[str(d)][str(h)]
            for x in range(x_max):
                for y in range(y_max):
                    if (d,h,x,y) in sample:
                        eye = windeye_search(data, x, y, width)
                        dist = (x-eye[0])**2+(y-eye[1])**2
                        dist = np.sqrt(dist)
                        add = eye[2]
                        strength = eye[3]
                        feature.append([dist, add, strength])
    return feature
import time
def neighbor(base_weather, sample):
    # 第二类： 风速整体情况
    # width=3,5,7,9,15,30 平均风速(包括上一小时，这一小时以及下一小时的增量),风速最大的距离，风速最小的距离，风速增量，风速方差
    wd = [3,5,7,15]
    feature = np.zeros((16283172, 33), dtype=np.float16)
    x_max = 548
    y_max = 421
    local_wd = 3
    # find neighbor
    i = 0
    for d in range(1, 6):
        for h in tqdm(range(3, 21)):
            data = base_weather[str(d)][str(h)]
            for x in range(x_max):
                for y in range(y_max):
                    if (d, h, x, y) in sample:
                        temp = []
                        temp.append(np.float16(data[x][y]-15))
                        for w in wd:
                            # current wind
                            this = middle_block(data,x,y,w)
                            temp.extend(this)
                            # last hour wind
                            try:
                                lastdata = base_weather[str(d)][str(h-1)]
                                last = middle_block(lastdata,x,y,w)
                                temp.extend(last)
                            except:
                                #temp.extend([None]*6)
                                last = this
                                temp.extend(last)

                            # next hour wind
                            try:
                                next_data = base_weather[str(d)][str(h+1)]
                                next_ = middle_block(next_data,x,y,w)

                                temp.extend(next_)
                            except:
                                #temp.extend([None]*6)
                                next_ = this
                                temp.extend(next_)
                            temp.extend([next_[0]-this[0], this[0]-last[0]])

                        feature[i,:] = temp
                        i += 1

    return feature



def windshape(base_weather, sample):
    # 第三类： 局部气团形状
    # 1 八个方向的风速波浪
    # 2 中间高周围小，中间小周围高，中间小两边大，中间大两边小
    pass
    # maybe is not that important！

def windedge_neardanger(data,x,y):
    x_max = 548
    y_max = 421
    mounter = np.array((x,y))
    mindist = 30
    for width in range(3,39,4):
        r = int((width - 1) / 2)
        x_up = max(0, x - r)
        x_down = min(x_max, x + r + 1)
        y_left = max(0, y - r)
        y_right = min(y_max, y + r + 1)
        block = data[x_up:x_down, y_left:y_right]
        if np.any(block>=15):
            block = np.array(block)
            index = np.argwhere(block>=15)
            # set default dist 80 if none of them bigger than 15
            for item in index:
                x_real = item[0]+x_up
                y_real = item[1]+y_left
                thispoint = np.array((x_real,y_real))
                dangerdist = np.linalg.norm(mounter - thispoint)
                mindist = min(mindist,dangerdist)
            break
    return np.float16(mindist)


def windedge_dangernum(data, x, y):
    x_max = 548
    y_max = 421
    temp = []
    for width in [9,15]:
        r = int((width - 1) / 2)
        x_up = max(0, x - r)
        x_down = min(x_max, x + r + 1)
        y_left = max(0, y - r)
        y_right = min(y_max, y + r + 1)
        block = data[x_up:x_down, y_left:y_right]
        num = len(np.where(block >= 15)[0])
        all_num = np.shape(block)[0]*np.shape(block)[1]
        rate = int(num)/all_num
        assert rate<=1
        temp.append(rate)
    return temp

def windedge(base_weather, sample):
    # 第四类: 设置边缘信息特征
    # t=-1,0,1的最近边缘信息,时间上最近的危险点，t=-1,0,1,w=5,w=9周围边缘点个数
    feature = []
    feature = np.zeros((16283172, 9), dtype=np.float16)

    x_max = 548
    y_max = 421
    local_wd = 3
    # find neighbor
    i = 0
    for d in range(1, 6):
        for h in tqdm(range(3, 21)):
            data = base_weather[str(d)][str(h)]
            for x in range(x_max):
                for y in range(y_max):
                    if (d, h, x, y) in sample :
                        temp = []
                        # find the nearest danger！
                        temp.append(windedge_neardanger(data, x, y))
                        # # find nearest danger in time
                        # windbyhour = np.array([base_weather[str(d)][str(k)][x][y] for k in range(3,21)])
                        # hourdist = 17
                        # if np.any(windbyhour >= 15):
                        #     indexbyhour = np.argwhere(windbyhour >= 15)
                        #     for item in indexbyhour:
                        #         hourdist = min(hourdist,abs(item[0]-h+3))
                        #     temp.append(int(hourdist))
                        # else:
                        #     temp.append(int(hourdist))
                        # last hour wind
                        try:
                            lastdata = base_weather[str(d)][str(h - 1)]
                            temp.append(windedge_neardanger(lastdata, x, y))
                        except:
                            #temp.append(None)
                            lastdata = data
                            temp.append(windedge_neardanger(lastdata, x, y))
                        # next hour wind
                        try:
                            next_data = base_weather[str(d)][str(h + 1)]
                            temp.append(windedge_neardanger(next_data, x, y))
                        except:
                            #temp.append(None)
                            next_data = data
                            temp.append(windedge_neardanger(next_data, x, y))
                        # find danger point num
                        [num9wd,num15wd] = windedge_dangernum(data, x, y)

                        temp.append(num9wd)
                        temp.append(num15wd)
                        # last hour
                        try:
                            lastdata = base_weather[str(d)][str(h - 1)]
                            [last_9wd,last_15wd] = windedge_dangernum(lastdata,x,y)

                            temp.append(num9wd-last_9wd)
                            temp.append(num15wd-last_15wd)
                        except:
                            #temp.append(None)
                            #temp.append(None)
                            lastdata = data
                            [last_9wd,last_15wd] = windedge_dangernum(lastdata, x, y)

                            temp.append(num9wd - last_9wd)
                            temp.append(num15wd - last_15wd)
                        # next hour
                        try:
                            nextdata = base_weather[str(d)][str(h + 1)]
                            [next_9wd,next_15wd] = windedge_dangernum(nextdata,x,y)

                            temp.append(next_9wd-num9wd)
                            temp.append(next_15wd-num15wd)

                        except:
                            #temp.append(None)
                            #temp.append(None)
                            nextdata = data
                            [next_9wd,next_15wd] = windedge_dangernum(nextdata, x, y)

                            temp.append(next_9wd - num9wd)
                            temp.append(next_15wd-num15wd)
                        #assert len(temp)==13
                        #print(len(temp))
                        feature[i,:] = temp
                        i += 1
    return feature

def rainedge_neardanger(data,x,y):
    x_max = 548
    y_max = 421
    mounter = np.array((x,y))
    mindist = 30
    for width in range(3,39,4):
        r = int((width - 1) / 2)
        x_up = max(0, x - r)
        x_down = min(x_max, x + r + 1)
        y_left = max(0, y - r)
        y_right = min(y_max, y + r + 1)
        block = data[x_up:x_down, y_left:y_right]
        if np.any(block>=4):
            block = np.array(block)
            index = np.argwhere(block>=4)
            # set default dist 80 if none of them bigger than 15
            for item in index:
                x_real = item[0]+x_up
                y_real = item[1]+y_left
                thispoint = np.array((x_real,y_real))
                dangerdist = np.linalg.norm(mounter - thispoint)
                mindist = min(mindist,dangerdist)
            break
    return np.float16(mindist)

def rainedge_dangernum(data, x, y):
    x_max = 548
    y_max = 421
    temp = []
    for width in [15]:
        r = int((width - 1) / 2)
        x_up = max(0, x - r)
        x_down = min(x_max, x + r + 1)
        y_left = max(0, y - r)
        y_right = min(y_max, y + r + 1)
        block = data[x_up:x_down, y_left:y_right]
        num = len(np.where(block >= 4)[0])
        all_num = np.shape(block)[0]*np.shape(block)[1]
        rate = int(num)/all_num
        assert rate<=1
        temp.append(rate)
    return temp

def rainedge(base_weather, sample):
    # 第四类: 设置边缘信息特征
    # t=-1,0,1的最近边缘信息,时间上最近的危险点，t=-1,0,1,w=5,w=9周围边缘点个数
    feature = []
    feature = np.zeros((16283172, 6), dtype=np.float16)

    x_max = 548
    y_max = 421
    local_wd = 3
    # find neighbor
    i = 0
    for d in range(1, 6):
        for h in tqdm(range(3, 21)):
            data = base_weather[str(d)][str(h)]
            for x in range(x_max):
                for y in range(y_max):
                    if (d, h, x, y) in sample :
                        temp = []
                        # find the nearest danger！
                        temp.append(rainedge_neardanger(data, x, y))
                        # last hour wind
                        try:
                            lastdata = base_weather[str(d)][str(h - 1)]
                            temp.append(rainedge_neardanger(lastdata, x, y))
                        except:
                            #temp.append(None)
                            lastdata = data
                            temp.append(rainedge_neardanger(lastdata, x, y))
                        # next hour wind
                        try:
                            next_data = base_weather[str(d)][str(h + 1)]
                            temp.append(rainedge_neardanger(next_data, x, y))
                        except:
                            #temp.append(None)
                            next_data = data
                            temp.append(rainedge_neardanger(next_data, x, y))
                        # find danger point num
                        [num15wd] = rainedge_dangernum(data, x, y)
                        # temp.append(num5wd)
                        # temp.append(num9wd)
                        temp.append(num15wd)
                        # last hour
                        try:
                            lastdata = base_weather[str(d)][str(h - 1)]
                            [last_15wd] = rainedge_dangernum(lastdata,x,y)
                            # temp.append(num5wd-last_5wd)
                            # temp.append(num9wd-last_9wd)
                            temp.append(num15wd-last_15wd)
                        except:
                            #temp.append(None)
                            #temp.append(None)
                            lastdata = data
                            [last_15wd] = rainedge_dangernum(lastdata, x, y)
                            # temp.append(num5wd - last_5wd)
                            # temp.append(num9wd - last_9wd)
                            temp.append(num15wd - last_15wd)
                        # next hour
                        try:
                            nextdata = base_weather[str(d)][str(h + 1)]
                            [next_15wd] = rainedge_dangernum(nextdata,x,y)
                            # temp.append(next_5wd-num5wd)
                            # temp.append(next_9wd-num9wd)
                            temp.append(next_15wd-num15wd)

                        except:
                            #temp.append(None)
                            #temp.append(None)
                            nextdata = data
                            [next_15wd] = rainedge_dangernum(nextdata, x, y)
                            # temp.append(next_5wd - num5wd)
                            # temp.append(next_9wd - num9wd)
                            temp.append(next_15wd-num15wd)

                        feature[i,:] = temp
                        i += 1
    return feature


def get_label(output_weather):
    x_max = 548
    y_max = 421
    wind_y = []
    rainfall_y = []
    prob_y = []
    for day in range(1, 6):
        for h in tqdm(range(3, 21)):
            for x in range(x_max):
                for y in range(y_max):
                    wind_y.append(output_weather[str(day)][str(h)][0, x, y])
                    rainfall_y.append(output_weather[str(day)][str(h)][1, x, y])
                    if output_weather[str(day)][str(h)][0, x, y] < 15 and output_weather[str(day)][str(h)][1, x, y] < 4:
                        prob_y.append(1)
                    else:
                        prob_y.append(0)
    return wind_y, rainfall_y, prob_y


def rainfall_neighbor(base_weather, sample):
    # 第二类： 风速整体情况
    # width=3,5,7,9,15,30 平均风速(包括上一小时，这一小时以及下一小时的增量),风速最大的距离，风速最小的距离，风速增量，风速方差
    wd = [3,5,7,15]
    feature = np.zeros((16283172, 33), dtype=np.float16)
    x_max = 548
    y_max = 421
    local_wd = 3
    # find neighbor
    i = 0
    for d in range(1, 6):
        for h in tqdm(range(3, 21)):
            data = base_weather[str(d)][str(h)]
            for x in range(x_max):
                for y in range(y_max):
                    if (d, h, x, y) in sample:
                        temp = []
                        temp.append(np.float16(data[x][y]-4))
                        for w in wd:
                            # current wind
                            this = middle_block(data, x, y, w)
                            temp.extend(this)
                            # last hour wind
                            try:
                                lastdata = base_weather[str(d)][str(h - 1)]
                                last = middle_block(lastdata, x, y, w)
                                temp.extend(last)
                            except:
                                # temp.extend([None]*6)
                                last = this
                                temp.extend(last)

                            # next hour wind
                            try:
                                next_data = base_weather[str(d)][str(h + 1)]
                                next_ = middle_block(next_data, x, y, w)
                                temp.extend(next_)
                            except:
                                # temp.extend([None]*6)
                                next_ = this
                                temp.extend(next_)
                            temp.extend([next_[0] - this[0], this[0] - last[0]])


                        feature[i,:] = temp
                        i += 1

    return feature


if __name__ == '__main__':

    base_weather = open('../layer_1_wind_shuffle.pkl', 'rb')
    base_weather = pickle.load(base_weather)
    #
    # base_weather = open('../layer_1_rainfall_shuffle.pkl', 'rb')
    # base_weather = pickle.load(base_weather)
    sample = pickle.load(open('sample.pkl', 'rb'))

    # 开始建立所有样本的周边特征 #####################

    # 第0类，确定每个样本点的气团流速，台风一般也就20km每小时

    # 第一类: 风眼特征
    # 风眼离我的距离，风眼相对我的速度增量,风眼速度
    # feature_windeye = windeye(base_weather, sample)
    # np.save('layer_2_feature_windeye_shuffle.npy', feature_windeye)

    # 第二类： 风速整体情况
    # width=3,5,7,9,15,30 平均风速(包括上一小时，这一小时以及下一小时的增量),风速最大的距离，风速最小的距离，风速增量，风速方差

    # feature_neighbor = neighbor(base_weather, sample)
    # np.save('layer_2_feature_wind_neighbor_shuffle.npy',feature_neighbor)

    # np.save('layer_2_add_negigbor.npy',add_neighbor(base_weather,sample))

    # 第三类： 局部气团形状
    # 1 八个方向的风速波浪
    # 2 中间高周围小，中间小周围高，中间小两边大，中间大两边小
    #feature_windshape = windshape(base_weather, sample)
    #feature_windshape = np.array(feature_windshape)
    #np.save('feature_windshape.npy', feature_windshape)

    #第四类: 设置边缘信息特征
    #最近边缘信息，周围边缘点个数
    # feature_windedge = windedge(base_weather, sample)
    # np.save('layer_2_feature_windedge_shuffle.npy', feature_windedge)

    # 第五类: 自身风速方差
    # t=-1,0,1的10个模型的风速方差及时间
    # basicinfo = np.load('../layer_1_basicinfo_129.npy')
    # index = np.load('sample_index.npy')
    # feature_basicinfo = basicinfo[index]
    # np.save('layer_2_feature_basicinfo.npy',feature_basicinfo)

    # rainfall neighbor
    # base_weather = open('../layer_1_rainfall_shuffle.pkl', 'rb')
    # base_weather = pickle.load(base_weather)
    # sample = pickle.load(open('sample.pkl', 'rb'))
    # rainfall = rainfall_neighbor(base_weather, sample)
    # np.save('layer_2_feature_rainfall_neighbor_shuffle.npy', rainfall)

    # 最近边缘信息，周围边缘点个数
    # base_weather = open('../layer_1_rainfall_shuffle.pkl', 'rb')
    # base_weather = pickle.load(base_weather)
    # sample = pickle.load(open('sample.pkl', 'rb'))
    # feature_windedge = rainedge(base_weather, sample)
    # np.save('layer_2_feature_rainedge_shuffle.npy', feature_windedge)

    # 样本标签
    # wind_y, rainfall_y, prob_y = get_label(pickle.load(open('weather_train_label_1_5.pkl', 'rb')))
    # index = np.load('sample_index.npy')
    # np.save('layer_2_label.npy',np.array(prob_y,np.float16)[index])


    # 收集所有特征和标签
    # cellect data

    # # merge the feature
    feature_wind_neighbor = np.load('layer_2_feature_wind_neighbor_shuffle.npy')
    feature_rainfall_neighbor = np.load('layer_2_feature_rainfall_neighbor_shuffle.npy')
    basicinfo = np.load('layer_2_feature_basicinfo.npy')
    windeye = np.array(np.load('layer_2_feature_windeye_shuffle.npy'),dtype=np.float16)
    windedge = np.load('layer_2_feature_windedge_shuffle.npy')
    rainedge = np.load('layer_2_feature_rainedge_shuffle.npy')
    X_train = np.concatenate((feature_wind_neighbor,feature_rainfall_neighbor,basicinfo,windeye,windedge,rainedge), axis=1)
    print(X_train.dtype)
    basicinfo,windeye,windedge = 0,0,0
    #X_train=np.array(X_train,dtype=np.float16)
    print(np.shape(X_train))
    np.save('layer_2_X_final.npy',X_train)
    #y=np.load('layer_2_label.npy')


    '''
    #split data
    #X_train = pd.read_csv('X.csv',header=None)
    y_train = np.load('label.npy')
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=1729)
    X_train.to_csv('X_train.csv', index=False, header=False)
    np.save('y_train.npy', y_train)
    X_test.to_csv('X_test.csv', index=False, header=False)
    np.save('y_test.npy', y_test)
    
    #spilit data 2
    #split data
    train_i = 2000000
    sample = 0
    X = pd.read_csv('X.csv',header=None)
    X_train = X.iloc[train_i:,:]

    #train_i = spilit_data(sample)
    #train_i = 8785138

    #feature_neighbor = pd.read_csv('feature_neighbor.csv',header=None) #91
    #basicinfo = pd.DataFrame(np.load('feature_basicinfo.npy')) #43
    #windeye = pd.DataFrame(np.load('feature_windeye.npy')) #3
    #windedge = pd.DataFrame(np.load('feature_windedge.npy')) #10
    #X_train = pd.concat([feature_neighbor,basicinfo,windeye,windedge], axis=1).iloc[train_i:,:]

    basicinfo,windeye,windedge = 0,0,0
    X_train.to_csv('X_train_v3_from_day1.csv', index=False, header=False)
    y_train = np.load('label.npy')[train_i:]
    np.save('y_train_v3_from_day1.npy', y_train)
    print('ceshiji')
    #X_test = pd.concat([feature_neighbor,basicinfo,windeye,windedge], axis=1).iloc[train_i:,:]
    X_test = X.iloc[0:train_i,:]
    X_test.to_csv('X_test_v3_from_day1.csv', index=False, header=False)
    y_test = np.load('label.npy')[0:train_i]
    np.save('y_test_v3_from_day1.npy', y_test)
    '''
    '''
    '''
    '''

    feature_neighbor = pd.read_csv('feature_neighbor.csv', header=None)  # 91
    basicinfo = pd.DataFrame(np.load('feature_basicinfo.npy'))  # 43
    windeye = pd.DataFrame(np.load('feature_windeye.npy'))  # 3
    windedge = pd.DataFrame(np.load('feature_windedge.npy'))  # 10
    # X_train = pd.concat([feature_neighbor,basicinfo,windeye,windedge], axis=1).iloc[0:train_i,:]
    # basicinfo,windeye,windedge = 0,0,0
    # X_train.to_csv('X_train_v2.csv', index=False, header=False)
    # y_train = np.load('label.npy')[0:train_i]
    # np.save('y_train_v2.npy', y_train)
    X_test = pd.concat([feature_neighbor, basicinfo, windeye, windedge], axis=1).to_csv('X.csv', index=False, header=False)
    y_test = np.load('label.npy')
    np.save('y.npy', y_test)
    '''

    '''
    
    # 生成libsvm txt format

    X_train = pd.read_csv('X_train.csv', header=None)
    y_train = np.load('y_train.npy')

    train_output = open('train.txt', 'w')
    ltrain = len(y_train)
    wtrain = len(X_train.columns)
    for i in tqdm(range(ltrain)):
        output_line = str(y_train[i])
        for j in range(wtrain):
            if pd.isnull(X_train.iloc[i, j]):
                pass
            else:
                output_line = output_line + ' ' + str(j + 1) + ':' + str(X_train.iloc[i, j])
        output_line = output_line + '\n'
        train_output.write(output_line)
    train_output.close()



    X_test = pd.read_csv('X_test.csv', header=None)
    y_test = np.load('y_test.npy')
    test_output = open('test.txt', 'w')
    ltest = len(y_test)
    wtest = len(X_test.columns)
    for i in tqdm(range(ltest)):
        output_line = str(y_test[i])
        for j in range(wtest):
            if pd.isnull(X_test.iloc[i, j]):
                continue
            else:
                output_line = output_line + ' ' + str(j + 1) + ':' + str(X_test.iloc[i, j])
        output_line = output_line + '\n'
        test_output.write(output_line)
    test_output.close()
    '''

