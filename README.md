# Introduction

- The main use of the following two papers "**slide window flitering**" and "**colorization using optimization**", for the above code, the implementation of the use of **Python** methods as well as the use of **Laplace operator optimization**(strengthen the edge and reduce the effect of the coloring bleeding), the experimental comparison results do have a certain effect of enhancement. So finally, the Laplace operator optimization is adopted.
- **colorzationOptimization.py** is a reproduction of the original paper on colorization using optimization, which is mainly to solve the system of linear equations in YUV space with the help of the theory of the above paper.
- **slideWindowColorization.py** is a reproduction of side window filtering + Laplace enhancement to achieve colorization filling.

# Usage

- ```
  python slideWindowColorization.py --padding 2 --gary_photo_file ./data/original/example1.png --marked_photo_file  ./data/marked/example1_marked.png --is_file # choose the single photo to colorizate
  python slideWindowColorization.py --padding 2 --gray_data_dir ./data/original --marked_data_dir ./data/marked # choose all files of folder to colorizate
  ```

## Parameter introduction

- | paramter          | content                                                      |
  | ----------------- | ------------------------------------------------------------ |
  | padding           | Side window window radius size, default is 2                 |
  | gary_photo_file   | When setting --is_file you need to set the gray scale image name |
  | marked_photo_file | Marked image name is required when setting --is_file         |
  | gray_data_dir     | Default setting is to read gray image folder path            |
  | marked_data_dir   | Default setting is to read marked image folder path          |
  | is_file           | False by default                                             |
  | exp_dir           | Defaults to . /data/exp                                      |
  | is_reshape        | Defaults to False                                            |
  | is_store          | when --is_store is in parameters list, the data should be stored in exp_dir, and its default setting is  false |

# Note

- When doing RGB to YUV space, it was found that dividing the RGB value by 255 works better than the maximum value of the enhanced RGB  when normalizing RGB.
- Links to specific papers as well as references are shown below.
  - [Side Window Filtering](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yin_Side_Window_Filtering_CVPR_2019_paper.pdf)
  - [Colorization Using Optimization](https://homepages.inf.ed.ac.uk/ksubr/Files/Papers/p689-levin.pdf)
  - [在侧窗滤波的框架下实现图像彩色化（超详细！！！附可执行代码）](https://blog.csdn.net/qq_52300384/article/details/128322428)

