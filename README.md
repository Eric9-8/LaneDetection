# LaneDetection

--_mIoU.py # 主要测试脚本*
--CULane_dataset_random.py
--do_main.py # 主要执行脚本*
--lr_scheduler.py # 主要warmup学习率策略*
--randomtest.py

--list # 包含了分割好的set.txt数据集和用于训练的gt_do_adam.txt数据集，可用于替换原始CULane数据集中的list文件夹

--Active_learning
     ----random_split.py # 划分为44个set.txt数据集
     ----Cdataset.py # AL读取set.txt生成数据集
     ----calculate.py 

--Student
     ----new_log
     ----new_log_v3 # v3表示是采用adam以及adamw优化，之前使用SGD
     ----utils
            ----transforms # 图像增广
	     ----data_augmentation.py
	     ----transforms.py
            ----CULane_dataset.py # 主要数据集读取
            ----lr_scheduler.py # 之前的warmup学习率策略
            ----my_regnet.py # 设置regnet中卷积为空洞卷积
            ----my_unet.py # student设置的unet结构(kd和student相同)
     ----config.json 
     ----model.py # student模型
     ----model_KD.py # 蒸馏(KD)模型
     ----train.py # 单独训练student模型
     ----train_kd.py # 单独训练kd模型

--Teacher
     ----new_log
     ----new_log_v3
     ----config.json
     ----model.py # teacher模型
     ----train.py # 单独训练teacher模型
     ----unet_T.py # teacher设置的unet结构
