# TianchiDataChallenge
## 懒得整理了，这次比赛的核心代码全部上传在这里。主要思路如下
1. 数据抽样 -> Sample data.ipynb 
2. EDA ->EDA.ipynb
3. StackNET 第一层 ->layer_1_model_gbm.ipynb
  主要是提取一些基础特征做回归，lightgbm模型速度比较快。
4. 第二层 ->layer_2.ipynb
5. 第三层 ->layer_3_model_1.ipynb
6. 第一层回归的天气数据进行特征提取，详见 ->feature_engineering.py
7. 天气预测结束后进行最优路线规划 ->routing_max_value_0105_multiprocessing_FINAL.py
8. 对算法计算的结果做可视化分析 ->result_analysis.ipynb
9. pytorch版本对天气进行预测的尝试 -> weather_prediction_pytorch.py
