## 单个代码程序说明

* [calcAccofEachCate.py](https://github.com/NewCoderQ/CataractClassification/blob/master/src-code/calcAccofEachCate.py) 分别计算每个类别的准确度
* [extractColorFeature.py](https://github.com/NewCoderQ/CataractClassification/blob/master/src-code/extractColorFeature.py) 提取图像的颜色特征，并且将其保存在pkl文件中
* [extractGLCMFeature.py](https://github.com/NewCoderQ/CataractClassification/blob/master/src-code/extractGLCMFeature.py) 提取图像的纹理特征，并且将其保存在pkl文件中
* [extractWaveFeature.py](https://github.com/NewCoderQ/CataractClassification/blob/master/src-code/extractWaveFeature.py) 提取图像的小波特征，并且将其保存在pkl文件中
* [main_extractFeatures.py](https://github.com/NewCoderQ/CataractClassification/blob/master/src-code/main_extractFeatures.py) 提取所有训练图像的三个特征
* [main_extractPredictFeatures.py](https://github.com/NewCoderQ/CataractClassification/blob/master/src-code/main_extractPredictFeatures.py) 提取所有测试集图像的三个特征
* [preProcess.py](https://github.com/NewCoderQ/CataractClassification/blob/master/src-code/preProcess.py) 对图像进行一些预处理，在这里就是对图像进行分割