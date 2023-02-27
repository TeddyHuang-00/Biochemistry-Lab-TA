- 2023/02/27：版本更新的 GST 数据收集和处理的程序已经迁移至[新仓库](https://github.com/TeddyHuang-00/GST-open-data-collect)，因此本仓库将会被归档
- 2023/02/27: An updated version of the contained app for collecting GST data and post-processing has been migrated to a [new repository](https://github.com/TeddyHuang-00/GST-open-data-collect), therefore this repository will be archived.

---

# Biochemistry-Lab-TA 生物化学实验助教

仅供教学或学习参考

## 部署需求

- 带公网 IP 的服务器
- Python3.6+

### 安装依赖

```sh
pip3 install numpy scipy pandas matplotlib streamlit
```

### 启动服务端实例

```sh
streamlit run GST.py --server.port=<Server Port>
```

## 提交数据

浏览器访问 `<IP>:<Server Port>` 即可，数据将罗列于 `data` 文件夹下（与 [GST.py](./GST.py) 同一路径）

## 整理数据

运行 [summarise_data.py](./summarise_data.py)

## 统计分析

见 [Analysis.ipynb](./Analysis.ipynb) (Python) 或 [hypo_test.pdf](./hypo_test.pdf) (R)

## 授课课件

见 [PPT](./课件.pptx) 或 [PDF](./课件.pdf)
