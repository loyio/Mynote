## Tensorflow 第一个项目

首先打开编辑器，Spyder或者是Pycharm等等

> 确保是安装过Tensorflow环境的



写下以下代码

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 08:08:40 2018

@author: susmote
"""
from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("Tensorflow version : {}".format(tf.VERSION))
print("Eager execution : {}".format(tf.executing_eagerly()))
```

最后输出结果

```bash
Tensorflow version : 1.7.0
Eager execution : True
```

