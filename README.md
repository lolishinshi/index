# index

基于特征点匹配的相似图片搜索工具

## 安装

**CPU 版**

```shell
pip install git+https://github.com/lolishinshi/index
```

**GPU 版（只有训练阶段需要）**

（请先安装 [anaconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)）

```shell
conda create -n index
conda activate index
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install -r requirements.gpu.txt
```

## 用法

### 导入图片

```shell
# 使用 16 线程
index add -t 16 /mnt/pictures
```

由于 Python 多进程效率限制，推荐线程数为操作系统线程数的一半

图片导入完毕后会在数据库目录（默认为 index.db）下创建两个 sqlite 数据库：
- metadata.db - 包含了图片的哈希和路径等信息
- vector.db - 包含了图片的特征点信息，该数据库在索引构建完毕后可以删除

### 训练索引

预估添加 2M 张图片，并以此为基准训练索引。

```shell
index train -n 2000000 --gpu
```

训练完毕后会在数据库目录下生成 `BIVF{K}_HNSW32.train` 文件，K 为聚类时划分的桶个数，由 n 计算得来。
以 2M 张图片为例，会生成 `BIVF1048576_HNSW32.train`。

训练时默认每个桶使用 50 个特征点训练，也就是 K/10 张图片。如果图片数量不足会影响训练效果，如果图片数量过多，则会延长训练时间。你可以通过 `-x 100` 来使用更多的图片训练。


### 构建索引

使用 BIVF1048576_HNSW32 作为模板，构建一个名为 image 的索引。完成会生成名为 `BIVF1048576_HNSW32.index.image` 的索引文件。

```shell
index build -d BIVF1048576_HNSW32 -n image
```

### 搜索

直接搜索本地图片

```shell
index search -n image test.jpg
```

提供 HTTP API

```shell
# 使用 mmap 减少内存占用
index server -n image --mmap
```

你可以在 `/docs` 路径下查看 API 文档

## TODO

- [x] 向量单独存放
- [ ] 分段索引
- [x] 分次添加索引
- [x] 图片去重
- [x] HTTP API
- [x] 图片应该缩小到屏幕大小，这样符合大多数人的阅读习惯
- [ ] 分布式多索引搜索
  - [ ] 搜索到足够准确的结果就可以返回
- [ ] 由于向量维度已经确定，可以使用 SIMD 优化 Hamming 距离的计算
- [ ] 添加图片时，多线程和多进程区别大不大（如何更好利用 CPU 资源）
- [ ] mask 掉文字部分

