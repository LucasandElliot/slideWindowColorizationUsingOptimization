# 介绍

- 主要是运用以下两篇论文《slide window flitering》和《colorization using optimization》，对于上述代码，实现运用的是Python方法以及使用Laplace算子进行优化，实验对比结果确实有一定提升效果。所以最后采取的是Laplace算子优化。
- colorzationOptimization.py为复现colorization using optimization的原论文
- slideWindowColorization.py为复现侧窗滤波+Laplace强化实现彩色化填充。

# 使用方法

- ```
  python slideWindowColorization.py --padding 2 --gary_photo_file ./data/original/example1.png --marked_photo_file  ./data/marked/example1_marked.png --is_file # 这里是选择单张图像处理
  python slideWindowColorization.py --padding 2 --gray_data_dir ./data/original --marked_data_dir ./data/marked # 这里是选择图像文件夹
  ```

## 参数说明

- | 参数                                | 内容 |
  | ----------------------------------- | ---- |
  |padding| 侧窗窗口半径大小，默认为2                 |
  |gary_photo_file|当设置--is_file的时候需要设置灰度图像名称|
  |marked_photo_file|当设置--is_file的时候需要设置标记图像名称|
  |gray_data_dir|默认设置为读取灰度图像文件夹路径|
  |marked_data_dir|默认设置为读取标记图像文件夹路径|
  |is_file|默认为False|
  |exp_dir|默认为./data/exp|
  |is_reshape|是否需要规格化图像大小|
  |is_store|是否需要存储在exp_dir之中，默认为False|

# 备注

- 在进行RGB转YUV空间的时候发现，将RGB规格化的时候，RGB数值除以255效果会比增强后的RGB的最大值【np.max(src)】效果好。
- 具体论文链接以及参考文献如下所示。
  - [Side Window Filtering](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yin_Side_Window_Filtering_CVPR_2019_paper.pdf)
  - [Colorization Using Optimization](https://homepages.inf.ed.ac.uk/ksubr/Files/Papers/p689-levin.pdf)
  - [在侧窗滤波的框架下实现图像彩色化（超详细！！！附可执行代码）](https://blog.csdn.net/qq_52300384/article/details/128322428)

