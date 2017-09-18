# cataract_demo.py demo测试文件的使用说明

### 该文件的作用是：根据用户输入的眼底图像ID，程序自动载入之前训练好的SVM模型，来对用户输入的眼底图像进行分类。(注意，此处笔者没有完成图形化界面的编程，感兴趣的同学可以自己尝试下，其实不是很困难。提示：可以使用tkinter模块)

## 使用方式
### 1、在normalization文件夹下新建一个show_data文件夹：
`mkdir show_data` 

该文件夹用来存储测试眼底图像的特征，以及整理完成的svm数据。
### 2、从github上更新本地服务器的代码文件：
`git pull` 

该命令在该项目文件夹下执行就可以，命令运行的结果就是可以从github远程仓库上更新代码文件到本地。
### 3、在normalization文件夹下执行以下命令来调用程序：
`python cataract_demo.py`

运行过程中会提示用户输入没有后缀名的眼底图像ID，测试图像数据被保存在父级目录的test/val文件夹下。
运行截图如下：
```
# python cataract_demo.py
Please enter the name of the image for classifying(without the ext):30375
提取图像的颜色特征：
30375.jpg
提取图像的纹理特征：
processing 1 / 1 / 1 image
processing 1 / 1 / 2 image
processing 1 / 1 / 3 image
processing 1 / 1 / 4 image
processing 1 / 1 / 5 image
processing 1 / 1 / 6 image
processing 1 / 1 / 7 image
processing 1 / 1 / 8 image
processing 1 / 1 / 9 image
processing 1 / 1 / 10 image
processing 1 / 1 / 11 image
processing 1 / 1 / 12 image
提取图像的小波特征：
1 / 1 / 1
1 / 1 / 2
1 / 1 / 3
1 / 1 / 4
1 / 1 / 5
1 / 1 / 6
1 / 1 / 7
1 / 1 / 8
1 / 1 / 9
1 / 1 / 10
1 / 1 / 11
1 / 1 / 12
图像的特征提取完成！！！
load test data...
generate SVM data...
test image classification...
model loading...
Accuracy = 0% (0/1) (classification)
重度白内障 眼底图像
```
程序运行完成之后，在最后一行就会输出最后的结果，此处为：重度白内障 眼底图像。
