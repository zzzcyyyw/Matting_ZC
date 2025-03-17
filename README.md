# matting_zc

## 权重文件说明
1. 本研究构建的前景对象的精细文本描述数据集已上传在：data/prompt.csv  
2. save_model/adapter/train_adapter_221.pth权重文件是stage1训练后得到的权重文件  
Depth_Anything_V2/checkpoints/depth_anything_v2_vitb.pth权重文件是用于得到深度图  
两个权重文件均大于100M，可能需使用git lfs相关命令进行拉取  
  
## 代码运行
整个代码运行分为了两个阶段：  
  
### stage1 训练depth adapter
深度图通过clip的图像编码器和编写的深度图适配器（depth adapter）转换为text embedding。  
1. 设置utils中的args.stage=1  
2. run train_depth_adapter_prompt.py  

### stage2 训练解码器
将RGB图通过clip的图像编码器和stage1得到的text embedding一起送入解码器进行训练。  
1. 设置utils中的args.stage=2  
2. run train_depth_adapter_prompt.py  
