整个程序在windows 10,  MATLAB R2016a 下完成的。

im2c.m : 用来得到Color Name 特征。
w2c.mat: im2c所需要的映射矩阵。
ColorName.m: 只用Color Name 特征的程序。
Hog.m:  只用HOG特征的程序。
hogCN:  结合Color Name 和 HOG特征的程序。

data1,data2: 6类数据集， 4类数据集
data1_flip, data2_flip：  扩充后的6类数据集， 4类数据集

data1.txt , data2.txt  data1_flip.txt , data2_flip.txt：存放文件夹目录，程序里会调用


PS：  在Color Name 和 hogCN 的程序里面， 降维之后的特征维数选择时候要根据情况手动设置的， 设置的数据在程序注释里。  另外要注意，测试的图片升成的特征维数要跟前面匹配。