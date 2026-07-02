`LET-Base-Dataset` 作为乐聚官方开源的大规模全尺寸人形机器人真机数据集，主要托管在国内外的几个主流开源AI社区和代码平台上。它没有一个单独域名的“独立官网”，而是以**官方开源项目主页**的形式存在。

可按网络环境与需求选用以下官方入口：

### 1. 国内最推荐：ModelScope（魔搭社区）主页

若主要**下载数据**（特别是真机 ROSbag 包），这里是国内的官方大本营，不限速，体验最好。

* **🔗 官方链接：** [ModelScope - lejurobot/LET-Base-Dataset](https://modelscope.cn/datasets/lejurobot/LET-Base-Dataset)
* **特点：** 详细列出了数据集的更新日志、数据规模（900多小时，74个任务目录）、多模态接口说明等，并且可以直接配合 `modelscope` 的 Python 库进行按需下载。

### 2. 代码与工程文档：GitCode / AtomGit 主页

文档说明、数据命名规范、传感器 Topic 映射与工具链见：

* **🔗 官方链接：** [GitCode - LET-Base-Dataset 开源主页](https://gitcode.com/lejurobot/LET-Base-Dataset) （或开放原子开源基金会的 [AtomGit 托管页](https://ai.atomgit.com/OpenLET/LET-Base-Dataset)）
* **特点：** 相当于该数据集的官方 Readme 现场。这里把夸父机器人（Kuavo 4 Pro / 5W）的 117 种原子技能编码、双臂/灵巧手/夹爪的控制 Topic 类型写得一清二楚。

### 3. 国际生态：Hugging Face 主页

若使用国际社区资源，或者后续想接入海外的一些具身智能大模型生态：

* **🔗 官方链接：** [Hugging Face - LejuRoboticsWS](https://huggingface.co/LejuRoboticsWS)
* **特点：** 乐聚机器人在 Hugging Face 上的官方组织账号，里面同样托管了 `LET-Base-Dataset` 以及专门针对灵巧手的 `LET-Dex-Dataset`。

---

### 💡 建议

可先收藏 **GitCode 主页** 查阅格式规范；在 **ModelScope** 用 Python 脚本按需下载数据集。