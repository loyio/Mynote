lris 分类器问题

> Iris数据集是常用的分类实验数据集，由Fisher, 1936收集整理。Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性。可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。

![Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Irisversicolor](..\images\iris_three_species.jpg)



#### 导入并解析训练数据集



##### 下载数据集

我们通过tf.keras.utils.get_file方法来获取数据集

代码如下

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 08:27:18 2018

@author: susmote
"""

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)

print("文件保存路径：{}".format(train_dataset_fp))
```



运行结果如下

```bash
Downloading data from http://download.tensorflow.org/data/iris_training.csv
8192/2194 [================================================================================================================] - 0s 0us/step
文件保存路径：C:\Users\susmote\.keras\datasets\iris_training.csv
```



将文件复制到当前工作路径





##### 检查数据

通过excel打开这个文件，可以简单了解这个数据的特征

![数据集](..\images\1524876582593.png)



通过对该数据集的分析，我们可以得出以下内容：

1. 第一行包含了这个数据集的有关信息
2. 这里总共有120个样例，一共有4个特征，每个样例都存在三种标签中的一种
3. 除去第一行后，前四列是特征，全部都是通过测量得到的，用浮点数保存
4. 最后一列，就是标签，是我们想要预测的值，对于这个数据集，他是用整数0,1,2代表的，对应一个花名



标签对应值

+ `0`  : Iris setosa
+ `1`  : Iris versicolor
+ `2`  : Iris virginica



##### 解析训练数据集

因为我们下载的是csv文件的数据集，所以我们可以定义一个函数去解析csv文件，在这个文件中，我们会调用tensorflow的方法以方便我们的使用，这个函数传入的参数是要解析的行数，返回的是特征和标签向量，功能是抓取前四个字段并将它们合并为一个张量，最后一个字段被解析为标签



代码如下

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:06:31 2018

@author: susmote
"""

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]] # 设置字段的类型
    parsed_line = tf.decode_csv(line, example_defaults) 
    feature = tf.reshape(parsed_line[:-1], shape=(4,)) # 前四个字段为特征，合并为一个张量
    label = tf.reshape(parsed_line[-1], shape=())   # 最后一个字段是标签
    return feature, label
```



##### 创建Tensorflow的训练集

- 前面已经定义了解析数据集的方法，现在我们要吧数据转换成Tensorflow的数据集，这样才能正常调用Tensorflow的方法，或者是API对数据进行更深一步的研究
- 在这里我们会使用`tf.data.TextLineDataset`方法来加载csv格式的数据文件，然后再调用我们定义好的方法`parse_csv`对数据进行解析
- 在这里，我们为了更快的训练模型，我们设置32个样本为一次训练



创建Tensorflow数据集对象的代码如下

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:22:23 2018

@author: susmote
"""

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from parse_dataset import parse_csv

train_dataset = tf.data.TextLineDataset('iris_training.csv')
train_dataset = train_dataset.skip(1)   # 跳过第一行的数据信息
train_dataset = train_dataset.map(parse_csv)    # 解析每一行
train_dataset = train_dataset.shuffle(buffer_size=1000)     # 打乱数据排列
train_dataset = train_dataset.batch(32)     # 每32行数据为一次训练

# 查看一个示例数据
features, label = tfe.Iterator(train_dataset).next()
print("示例特征 ：", features[0])
print("示例标签 ：", label[0])
```



运行结果如下

```bash
示例特征 ： tf.Tensor([ 6.19999981  2.79999995  4.80000019  1.79999995], shape=(4,), dtype=float32)
示例标签 ： tf.Tensor(2, shape=(), dtype=int32)
```





#### 建立一个模型

##### 为什么要建模

模型是特征和标签之间的关系，对于这个Iris分类问题，非常适合建立一个关系模型，一些简单的模型可以用几行代数来描述，但是复杂的机器学习模型有很多难以概括的参数，通过机器学习模型会很好的让你找出特征和标签之间的关系



##### 选择模型

机器学习有很多模型供我们去选择，在这里，对于这个Iris分类问题，我们选择神经网络(Neural networks)来解决这个问题



神经网络能够找到特征与标签中的复杂的关系

> 它是一个高度结构化的图，组织成一个或多个隐藏层。每个隐藏层由一个或多个神经元组成。有几种类型的神经网络，该程序使用密集的或全连接的神经网络：一层中的神经元接收来自前一层中每个神经元的输入连接。

例如，下面这张图说明了一个由输入层(Input Layer)，两个隐藏层(Hidden Layer)和一个输出层(Output Layer)组成的密集神经网络：

![A diagram of the network architecture: Inputs, 2 hidden layers, and outputs](..\images\full_network.png)

对于这张图的简单理解就是：首先我们训练了一个模型，然后我们把一个没有标签的样本放入这个模型中，通过其中的隐藏层对模型进行分析，最终会输出三个标签的预测值，通过对比预测值，从而得出标签值



##### 使用keras建立一个模型

Keras是一个高层神经网络API，Keras由纯Python编写而成并基[Tensorflow](https://github.com/tensorflow/tensorflow)、[Theano](https://github.com/Theano/Theano)以及[CNTK](https://github.com/Microsoft/cntk)后端。Keras 为支持快速实验而生，能够把你的idea迅速转换为结果

官方文档 ：https://keras.io/

中文文档 ：http://keras-cn.readthedocs.io/en/latest



*tf.keras.Sequential*模型是一个线性堆栈层。它的构造函数需要一个图层实例列表，在这种情况下，两个密集图层各有10个节点，输出图层有3个节点代表我们的标签预测。第一层的*input_shape*参数对应于数据集中要素的数量，并且是必需的。

```python
# -*- coding: utf-8 -*-
"""
Created on 2018/5/3 

@author: susmote
"""

import tensorflow as tf


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4, )),
    tf.keras.layers.Dense(10, activation="relu")
    tf.keras.layers.Dense(3)
])
```



#### 训练模型

>训练是当模型逐步优化或模型学习数据集时机器学习的阶段。我们的目标是充分了解训练数据集的结构，以预测未知数据。如果您对训练数据集了解太多，则预测仅适用于所看到的数据，并且不会推广。这个问题被称为过度拟合 - 就像记忆答案而不是理解如何解决问题。





##### 定义损失函数和梯度函数

> 训练和评估阶段都需要计算模型的损失。这可以衡量模型的预测来自期望的标签，换句话说，模型的表现有多糟糕。我们想要最小化或优化这个值



```python
# -*- coding: utf-8 -*-
"""
Created on 2018/5/3 

@author: susmote
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)
```



##### 创建优化器

优化程序将计算出的梯度应用于模型的变量以最小化损失函数。你可以想象一个曲面（见下图），我们希望通过走动找到最低点。梯度指向最陡峭的上升方向 ， 所以我们将以相反的方式行驶并沿着山坡下移。通过迭代计算每个批次的损失和梯度，我们将在训练期间调整模型。逐渐地，模型会找到权重和偏差的最佳组合，以最大限度地减少损失。损失越低，模型的预测就越好。

![Optimization algorthims visualized over time in 3D space.](../images/opt1.png)



> TensorFlow有许多可用于训练的优化算法。该模型使用实现随机梯度下降（SGD）算法的*tf.train.GradientDescentOptimizer*。 *learning_rate*为沿着山丘的每次迭代设置步长。这是一个超参数，您通常会调整以获得更好的结果。



```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
```



##### 循环训练

已经做好了前面的工作，现在就要正式开始进行模型的训练



接下来的代码会围绕下面这些训练步骤进行编写

1. 迭代每个时期，一个时期通过一次数据集
2. 在一个时期内，对训练的`Dataset`的每个示例进行迭代，以获取其特征（`x`）和标签（`y`）
3. 使用示例的功能，进行预测并将其与标签进行比较。测量预测的不准确性并使用它来计算模型的损失和梯度。
4. 使用优化器来更新变量
5. 对一些统计数据进行可视化操作
6. 重复每个时期



变量的说明：

- `num_epochs`	:	循环遍历数据集的次数 
  - num_epochs是一个可以调优的超参数。选择正确的数字通常需要经验和实验



训练代码如下

