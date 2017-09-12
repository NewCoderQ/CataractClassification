## 运行环境配置过程中的一些错误以及解决方式

### pip安装
此处不再赘述，需要确认的是：自己的pip是否和默认的python是同一个版本

确认方法：

```
~# which pip
/usr/local/bin/pip
~# vim /usr/local/bin/pip
```
出现以下内容：
```
#!/usr/bin/python3

# -*- coding: utf-8 -*-
import re
import sys

from pip import main

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())

```
检查第一行的python*是否和系统默认的python是同一个版本。

### numpy安装
命令：
`pip install numpy`

### opencv-python安装
命令：
`pip install opencv-python`

```	
>>> import cv2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/dist-packages/cv2/__init__.py", line 9, in <module>
    from .cv2 import *
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory

apt-get install libglib2.0-0:* 

>>> import cv2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/dist-packages/cv2/__init__.py", line 9, in <module>
    from .cv2 import *
ImportError: libSM.so.6: cannot open shared object file: No such file or directory

#apt-get install libSM*
```

以上就步骤就完成了基础动态链接库的安装。

`ImportError: No module named pywt`

执行以下命令从github上拷贝到本地

`root@8b62069d93d5:~/lib_install# git clone https://github.com/NewCoderQ/pywt`

执行下面的命令来安装pywt

	
上面的错误消息提示在安装PyWavelets之前需要安装Cython
执行下面这句命令来完成Cython的安装
`pip2 install Cython`

Cython完成之后，再次执行下面这条语句来安装pywt
`root@8b62069d93d5:~/lib_install/pywt# pip2 install .`

libsvn-python的配置和安装

* 首先下载[libsvm](https://github.com/cjlin1/libsvm)

```
$ git clone https://github.com/cjlin1/libsvm
$ cd libsvm
$ make
$ cd python
$ make
$ cp *.py /usr/lib/python2.7/dist-packages/  
$ cd ..  
$ cp libsvm.so.2 /usr/lib/python2.7/  

```
上述命令完成之后，用以下命令进行测试，查看是否安装成功：

```
python
import svm
import svmutil
```

如果没有任何错误信息，即安装成功。