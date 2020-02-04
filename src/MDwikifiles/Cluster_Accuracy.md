# 聚类精确度 (Accuracy, AC)


[知乎超链](https://zhuanlan.zhihu.com/p/53840697)

[匈牙利算计说明](https://www.programcreek.com/python/example/91391/sklearn.utils.linear_assignment_.linear_assignment)

聚类精确度 (Accuracy, AC) 用于比较获得标签和数据提供的真实标签:

$$
  AC = \frac{\Sigma^{n}_{i=1}\delta(s_i,map(r_i))}{n}   
$$

其中，$r_i$和$s_i$表示数据$x_i$所对应的获得的标签和真实标签，$n$为数据总的个数，$\delta$表示指数函数如下：

$$
  \delta(x,y) = \begin{cases}
        1 \text{ if } x = y \\
        0 \text{ otherwise } 
      \end{cases}
$$

而式中的$map$ 则表示最佳类标的重现分配，以才能保证统计的正确。一般的该最佳重分配可以通过匈牙利算法 (Kuhn-Munkres or Hungarian Algorithm) 实现，从而在多项式时间内求解该任务（标签）分配问题。

```python
  def acc(y_true, y_pred):
  """
  Calculate clustering accuracy. Require scikit-learn installed
  # Arguments
      y: true labels, numpy.array with shape `(n_samples,)`
      y_pred: predicted labels, numpy.array with shape `(n_samples,)`
  # Return
      accuracy, in [0,1]
  """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size 
```

**举例说明：**

```python
import numpy as np
def cluster_acc(Y_pred, Y):
  # 导入匈牙利算法所需要的sklearn的库 
  from sklearn.utils.linear_assignment_ import linear_assignment
  # 判断样本的个数是否相同
  assert Y_pred.size == Y.size
  # 计算最大标签的值
  D = max(Y_pred.max(), Y.max())+1
  # 创建标签转换度量矩阵
  w = np.zeros((D,D), dtype=np.int64)
  # 标签转换度量矩阵赋值
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  # 使用匈牙利算法求解最佳类标的重现分配
  ind = linear_assignment(w.max() - w)
  # 公式求解聚类精度(Cluster Accuracy)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

if __name__ == "__main__":
    # 真实标签
    Y = np.array([1,1,2,2,3,3])
    # 预测标签
    PY = np.array([2,2,3,1,1,3])
    # 聚类精度(Cluster Accuracy)结果
    result = cluster_acc(PY,Y)
    print(result)
```

## 代码总变量的变化

在上述计算代码中，标签转换度量矩阵w的值$w_1$为
$$
\begin{bmatrix}
   0 & 0 & 0 & 0 \\
   0 & 0 & 1 & 1 \\
   0 & 2 & 0 & 0 \\
   0 & 0 & 1 & 1 
\end{bmatrix}
 $$

 在喂入匈牙利算法求解前的矩阵换位w.max() - w的矩阵$w_2$为：
 $$
\begin{bmatrix}
   2 & 2 & 2 & 2 \\
   2 & 2 & 1 & 1 \\
   2 & 0 & 2 & 2 \\
   2 & 2 & 1 & 1 
\end{bmatrix}
 $$

匈牙利算法求解后ind的矩阵$\nu_1$为：
 $$
\begin{bmatrix}
   0 & 0 \\
   1 & 2 \\
   2 & 1  \\
   3 & 3 
\end{bmatrix}
 $$
其中$\nu_1$代表了将$lable_1$转换为$lable_2$,$lable_2$转换为$lable_1$,其它的标签不变。

## 匈牙利算法的作用

**问题：** 假定某单位有甲、乙、丙、丁、戊五个员工，现需要完成A、B、C、D、E五项任务，每个员工完成某项任务的时间如下图所示，应该如何分配任务，才能保证完成任务所需要的时间开销最小？![1](https://images2015.cnblogs.com/blog/1027162/201610/1027162-20161009210043274-862077955.png)

在求解过程中，即如上面的求解最小值问题，矩阵$w_2$即表示横向标签与纵向标签，重合度最高之间的关系。即$lable_1$与$lable_2$重合度较高，依此类推。
最后即得到最佳类标的重现分配的映射关系，如矩阵$\nu_1$。用此映射关系即可求解$\delta(s_i,map(r_i))$
