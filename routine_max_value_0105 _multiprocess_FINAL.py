import pandas as pd
import numpy as np
import time
import datetime
import pickle
from multiprocessing import Pool
from tqdm import tqdm
tqdm.monitor_interval = 0
#data_for_test = pd.read_csv('../ForecastDataforTesting_20171124.csv')

def find_the_path(param):
    id, weather_map, day, start_x, start_y, end_x, end_y, x_min, x_max, y_min, y_max = param
    # 超过15坠毁

    def evaluate_weather(x, y, point):
        weather = weather_map[day][str(point // 30 + 3)][x][y]
        return weather

    def calculate_value(p_after, point):
        return p_after*(2*point-1440)

    wind_block = 0.01
    stop = False

    # 绘制结束可选状态
    # 最小完成时间
    t_least = 2 * (abs(start_x - end_x) + abs(start_y - end_y)) // 60
    # 倒推最小路径
    # 获取可用路径
    t_available = []
    for t in range(3, 21):
        if t - 3 >= t_least and weather_map[day][str(t)][end_x][end_y] > wind_block:
            t_available.append(t)
    # 遍历关键节点路径
    if len(t_available) == 0:
        print('no end points available!')
        return 0
    elif len(t_available) == 1:
        pass
    else:
        temp = []
        for i in range(0, len(t_available) - 1):
            if t_available[i] - t_available[i + 1] == -1:
                continue
            else:
                temp.append(t_available[i])
        temp.append(t_available[-1])
        t_available = temp

    routes = []
    print(t_available)
    available_start = []
    for t in t_available:
        print(t)
        # 设置存储变量初始值
        stop = False
        t_start = (t-2)*30-1
        legal_state = {}
        value_max = {}
        legal_state[str(t_start)] = set()
        legal_state[str(t_start)].add((end_x, end_y))
        # p_after contains this point prob
        value_max[(end_x, end_y, t_start)] = {'max_value':calculate_value(evaluate_weather(end_x,end_y,t_start),t_start),
                                              'p_after':evaluate_weather(end_x,end_y,t_start),'end_point':t_start, 'routine':[(end_x,end_y,t_start)]}
        # 设置迭代程序
        for point in tqdm(range(0, t_start)[::-1]):
            #renew_state(legal_state[str(point + 1)], point)
            last_state = legal_state[str(point+1)]
            legal_state[str(point)] = set()
            for item in last_state:
                # 节约计算时间
                if abs(item[0] - start_x) + abs(item[1] - start_y) > point:  # 还有改进空间
                    continue
                new_state = [(item[0], item[1]), (item[0] + 1, item[1]), (item[0] - 1, item[1]), (item[0], item[1] + 1),
                             (item[0], item[1] - 1)]
                for next_state in new_state:
                    # 是否在指定区域内
                    if x_min <= next_state[0] <= x_max and y_min <= next_state[1] <= y_max:
                        # 之前是否保存过
                        if next_state not in legal_state[str(point)]:
                            # 天气是否合格
                            if weather_map[day][str(point // 30 + 3)][next_state[0]][next_state[1]] > wind_block:
                                legal_state[str(point)].add(next_state)
                                # 判断某个时刻的legal_state是否为空
            if len(legal_state[str(point)]) == 0:
                print(id, day, t, 'legal state is null')
                break
            for item in legal_state[str(point)]:
                if item[0] == end_x and item[1] == end_y:
                    wea_end = evaluate_weather(end_x,end_y,point)
                    value_max[(end_x, end_y, point)] = {'max_value':calculate_value(wea_end,point),'p_after': wea_end , 'end_point': point,'routine':[(end_x,end_y,point)]}
                else:
                    temp = []
                    old_state = [(item[0], item[1]), (item[0] + 1, item[1]), (item[0] - 1, item[1]),
                                 (item[0], item[1] + 1),
                                 (item[0], item[1] - 1)]
                    for o_s in old_state:
                        if o_s in last_state:
                            temp.append((value_max[(o_s[0], o_s[1], point + 1)]['max_value'],(o_s[0], o_s[1], point + 1)))

                    temp.sort(key=lambda x: x[0])

                    # 同一时间段同一个点
                    if item[0]==temp[0][1][0] and item[1]==temp[0][1][1] and point%30!=29:
                        current_node = [(item[0], item[1], point)]
                        current_node.extend(value_max[temp[0][1]]['routine'])
                        value_max[(item[0], item[1], point)] = {'max_value':temp[0][0],'p_after':value_max[temp[0][1]]['p_after'],
                                                                'end_point':value_max[temp[0][1]]['end_point'],'routine':current_node}
                    else:
                        #p = evaluate_weather(temp[0][1][0], temp[0][1][1], point)
                        p = evaluate_weather(item[0], item[1], point)
                        p_after = p*value_max[temp[0][1]]['p_after']
                        # min(1440 +(2*n-1440)*Fj*pj)
                        new_end_point = value_max[temp[0][1]]['end_point']
                        max_value = calculate_value(p_after, new_end_point)
                        current_node = [(item[0], item[1], point)]
                        current_node.extend(value_max[temp[0][1]]['routine'])
                        value_max[(item[0], item[1], point)] = {'max_value':max_value,'p_after':p_after,'end_point':new_end_point,'routine':current_node}
            # # release memory
            if point<t_start-4:
                del legal_state[str(point + 3)]
            #     for item in legal_state[str(point+2)]:
            #         del value_max[(item[0],item[1],point+2)]




                # return 0
        # if stop == True:
        #     continue

        # 寻找最优路径
        for k in range(0, t_start):
            try:
                available_start.append((k,value_max[(start_x,start_y,k)]['max_value'],value_max[(start_x,start_y,k)]['routine']))
            except:
                continue


    available_start.sort(key=lambda x:x[1])
    if len(available_start)==0:
        print('failed')
        return 0
    elif len(available_start)>30:
        print(id, int(day) + 5, 'success')
        return (id, int(day) + 5,available_start[0:30])
    else:
        print(id, int(day) + 5, 'success')
        return (id, int(day) + 5,available_start)


    #     route = []
    #     # 判断初始节点是否在legal_state中
    #     if (start_x, start_y) not in legal_state['0']:
    #         # print(id, day, ' 找不到最优路径，初始节点不在可行解上')
    #         continue
    #     else:
    #         route.append((start_x, start_y, 0))
    #         current_x = start_x
    #         current_y = start_y
    #         for t in range(1, t_start):
    #             item = [current_x, current_y]
    #             if current_x == end_x and current_y == end_y:
    #                 break
    #             potential_state = [(item[0], item[1]), (item[0] + 1, item[1]), (item[0] - 1, item[1]), (item[0], item[1] + 1), (item[0], item[1] - 1)]
    #             temp = []
    #             for next_state in potential_state:
    #                     if next_state in legal_state[str(t)]:
    #                        temp.append([next_state, value_max[(next_state[0], next_state[1], t)]['max_value']])
    #
    #             temp.sort(key=lambda x: x[1])
    #             current_x = temp[0][0][0]
    #             current_y = temp[0][0][1]
    #             route.append((current_x,current_y,t))
    #             if t == t_start - 1:
    #                 route.append((end_x, end_y, t_start))
    #             #legal_state[str(t)]
    #         #punish_total = value_max[(start_x, start_y, 0)]
    #     if len(route) != 0 :
    #         #
    #         routes.append((value_max[(start_x, start_y, 0)]['max_value'], route))
    #     else:
    #         continue
    #
    #
    # if len(routes) != 0 :
    #     route = sorted(routes, key=lambda student: student[0])[0]
    #     print(id,day, 'punish:', route[0])
    #     print(' 最优路径为: ', route[1])
    #     return (id, int(day) + 5, route[1])
    # else:
    #     print(id, int(day)+5)
    #     print('can not find it!')
    #     return 0


if __name__ == "__main__":
    print('程序开始：')
    citys = pd.read_csv('CityData.csv')
    #file_weather = open('weather_online_output_1227_h_p.pkl', 'rb')
    file_weather = open('layer_1_output_prob.pkl', 'rb')
    #file_weather = open('weather_guanyang.pkl', 'rb')
    weather_map = pickle.load(file_weather)
    # parameters for the solutions
    # 自然界的风在空间上是连续的，在时间上也是连续的。
    result = []
    params = []
    for day in range(6, 11):
        start_x = citys.loc[0, 'xid'] - 1
        start_y = citys.loc[0, 'yid'] - 1
        end_points = np.array(citys.loc[1::, :])
        x_max = 548 - 1
        x_min = 1 - 1
        y_max = 421 - 1
        y_min = 1 - 1
        # max_hour = 8
        # t_max = int(max_hour * 60 / 2)  # （18h）*（60min/h）/（2min/格）

        for dest in end_points:
            id = dest[0]
            end_x = dest[1]-1
            end_y = dest[2]-1
            
            params.append((id, weather_map, str(day-5), start_x, start_y, end_x, end_y, x_min, x_max, y_min, y_max))
    print(len(params))
    pool = Pool(3)
    #print(params)
    result = pool.map(find_the_path, params)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    #routes = find_the_path()
    #result.append([id, day, routes])
    print(len(result))
    # 制作提交文件格式
    submit = []
    base_time = "2017-12-10 03:00:00"
    base_time = datetime.datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")

    for item in result:
        if item != 0:
            time = base_time
            for point in item[2]:
                time_point = base_time+datetime.timedelta(minutes=2)*point[2]

                time_format = time_point.strftime('%H:%M')
                submit.append([item[0], item[1], time_format, point[0]+1, point[1]+1])
    file_data = pd.DataFrame(submit)
    print(file_data)
    file_data.to_csv('submit_file_0130.csv', index=False, header=False)
