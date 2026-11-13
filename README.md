# 原文库 (yuanwen)

把与 Claude 的聊天原文逐字保存在自己的服务器上，并通过 MCP 连接器做语义定向召回。

- 存储：SQLite 单文件，写入后不衰减、不合并、不改写
- 检索：语义向量 + 关键词混合，支持日期范围，逐字返回
- 工具：`recall`、`expand`、`get_conversation`、`list_conversations`、`save`

域名、连接密钥和称呼只保存在服务器本机的 `/opt/yuanwen`，不在本仓库中。

## 更新

在服务器上运行：

```bash
yuanwen-update
```
