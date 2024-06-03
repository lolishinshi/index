# indekkusu

- [x] 向量单独存放
- [ ] 分段索引
- [ ] 分次添加索引
- [ ] 图片去重
- [ ] HTTP API
- [x] 图片应该缩小到屏幕大小，这样符合大多数人的阅读习惯
- [ ] 分布式多索引搜索
  - [ ] 搜索到足够准确的结果就可以返回
- [ ] 由于向量维度已经确定，可以使用 SIMD 优化 Hamming 距离的计算

## 测试

- 34788 张图 x 500 特征点 x 256bit 特征向量
  - i9-13900HX + DDR5-5200
  - 理论空间占用 0.5GiB
  - index.db
    - leveldb 大小 507MiB
  - 索引
    - 耗时 6m16s
    - 峰值内存 4.6GiB
    - 大小 2.7GiB
  - 查询
    - 66ms
- 112418 张图 x 500 特征点 x 256bit 特征向量
  - 实际添加 108386 图
  - 3970X + DDR4-2666
  - index.db
    - image 大小 3.5MiB
    - vector 大小 1.5GiB
  - 索引
    - 耗时 27m30s
    - 峰值内存 15.1GiB
    - 大小 8.3GiB
  - 查询
    - 内存消耗 376MiB
    - 485ms

## Thanks

- https://github.com/unum-cloud
- https://github.com/BAILOOL/ANMS-Codes
- https://github.com/unum-cloud/usearch/issues/150#issuecomment-1826901704
