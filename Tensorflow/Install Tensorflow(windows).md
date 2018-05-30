## 安装Tensorflow（windows）

> TensorFlow是谷歌基于DistBelief进行研发的第二代人工智能学习系统，其命名来源于本身的运行原理。Tensor（张量）意味着N维数组，Flow（流）意味着基于数据流图的计算，TensorFlow为张量从图象的一端流动到另一端计算过程。TensorFlow是将复杂的数据结构传输至人工智能神经网中进行分析和处理过程的系统。



最近开始学习实战机器学习，所以就要用到一些工具来提高我的运算效率，而Tensorflow是一个非常好的选择，对于我去学习机器学习，深度学习这方面的内容



#### windows端安装Tensorflow非常简单

只需运行以下命令即可

```bash
pip install tensorflow
```



然后命令行提示以下选项

```bash
C:\Users\susmote>pip install tensorflow
Collecting tensorflow
  Downloading https://files.pythonhosted.org/packages/35/f6/8af765c7634bc72a902c50d6e7664cd1faac6128e7362510b0234d93c974/tensorflow-1.7.0-cp36-cp36m-win_amd64.whl (33.1MB)
    100% |████████████████████████████████| 33.1MB 46kB/s
Collecting protobuf>=3.4.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/32/cf/6945106da76db9b62d11b429aa4e062817523bb587018374c77f4b63200e/protobuf-3.5.2.post1-cp36-cp36m-win_amd64.whl (958kB)
    100% |████████████████████████████████| 962kB 62kB/s
Collecting grpcio>=1.8.6 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/80/7e/d5ee3ef92822b01e3a274230200baf2454faae64e3d7f436b093ff771a17/grpcio-1.11.0-cp36-cp36m-win_amd64.whl (1.4MB)
    100% |████████████████████████████████| 1.4MB 98kB/s
Requirement already satisfied: numpy>=1.13.3 in d:\users\susmote\anaconda3\lib\site-packages (from tensorflow)
Collecting absl-py>=0.1.6 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/90/6b/ba04a9fe6aefa56adafa6b9e0557b959e423c49950527139cb8651b0480b/absl-py-0.2.0.tar.gz (82kB)
    100% |████████████████████████████████| 92kB 87kB/s
Collecting termcolor>=1.1.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz
Requirement already satisfied: six>=1.10.0 in d:\users\susmote\anaconda3\lib\site-packages (from tensorflow)
Collecting tensorboard<1.8.0,>=1.7.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/0b/ec/65d4e8410038ca2a78c09034094403d231228d0ddcae7d470b223456e55d/tensorboard-1.7.0-py3-none-any.whl (3.1MB)
    100% |████████████████████████████████| 3.1MB 49kB/s
Collecting astor>=0.6.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/b2/91/cc9805f1ff7b49f620136b3a7ca26f6a1be2ed424606804b0fbcf499f712/astor-0.6.2-py2.py3-none-any.whl
Collecting gast>=0.2.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/5c/78/ff794fcae2ce8aa6323e789d1f8b3b7765f601e7702726f430e814822b96/gast-0.2.0.tar.gz
Requirement already satisfied: wheel>=0.26 in d:\users\susmote\anaconda3\lib\site-packages (from tensorflow)
Requirement already satisfied: setuptools in d:\users\susmote\anaconda3\lib\site-packages (from protobuf>=3.4.0->tensorflow)
Collecting markdown>=2.6.8 (from tensorboard<1.8.0,>=1.7.0->tensorflow)
  Downloading https://files.pythonhosted.org/packages/6d/7d/488b90f470b96531a3f5788cf12a93332f543dbab13c423a5e7ce96a0493/Markdown-2.6.11-py2.py3-none-any.whl (78kB)
    100% |████████████████████████████████| 81kB 65kB/s
Collecting html5lib==0.9999999 (from tensorboard<1.8.0,>=1.7.0->tensorflow)
  Downloading https://files.pythonhosted.org/packages/ae/ae/bcb60402c60932b32dfaf19bb53870b29eda2cd17551ba5639219fb5ebf9/html5lib-0.9999999.tar.gz (889kB)
    100% |████████████████████████████████| 890kB 44kB/s
Collecting bleach==1.5.0 (from tensorboard<1.8.0,>=1.7.0->tensorflow)
  Downloading https://files.pythonhosted.org/packages/33/70/86c5fec937ea4964184d4d6c4f0b9551564f821e1c3575907639036d9b90/bleach-1.5.0-py2.py3-none-any.whl
Requirement already satisfied: werkzeug>=0.11.10 in d:\users\susmote\anaconda3\lib\site-packages (from tensorboard<1.8.0,>=1.7.0->tensorflow)
Building wheels for collected packages: absl-py, termcolor, gast, html5lib
  Running setup.py bdist_wheel for absl-py ... done
  Stored in directory: C:\Users\susmote\AppData\Local\pip\Cache\wheels\23\35\1d\48c0a173ca38690dd8dfccfa47ffc750db48f8989ed898455c
  Running setup.py bdist_wheel for termcolor ... done
  Stored in directory: C:\Users\susmote\AppData\Local\pip\Cache\wheels\7c\06\54\bc84598ba1daf8f970247f550b175aaaee85f68b4b0c5ab2c6
  Running setup.py bdist_wheel for gast ... done
  Stored in directory: C:\Users\susmote\AppData\Local\pip\Cache\wheels\9a\1f\0e\3cde98113222b853e98fc0a8e9924480a3e25f1b4008cedb4f
  Running setup.py bdist_wheel for html5lib ... done
  Stored in directory: C:\Users\susmote\AppData\Local\pip\Cache\wheels\50\ae\f9\d2b189788efcf61d1ee0e36045476735c838898eef1cad6e29
Successfully built absl-py termcolor gast html5lib
Installing collected packages: protobuf, grpcio, absl-py, termcolor, markdown, html5lib, bleach, tensorboard, astor, gast, tensorflow
  Found existing installation: html5lib 0.999999999
    Uninstalling html5lib-0.999999999:
      Successfully uninstalled html5lib-0.999999999
  Found existing installation: bleach 2.0.0
    Uninstalling bleach-2.0.0:
      Successfully uninstalled bleach-2.0.0
Successfully installed absl-py-0.2.0 astor-0.6.2 bleach-1.5.0 gast-0.2.0 grpcio-1.11.0 html5lib-0.9999999 markdown-2.6.11 protobuf-3.5.2.post1 tensorboard-1.7.0 tensorflow-1.7.0 termcolor-1.1.0
```



#### 安装成功后，我们测试是否可用

首先进入到python交互器界面中

运行以下代码

```python
In [1]: import tensorflow as tf

In [2]: tf.__version__
Out[2]: '1.7.0'
```



这就证明windows下的Tensorflow成功安装



之后我们会通过tensorflow完成许多有趣的工作

