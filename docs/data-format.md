# 数据格式说明

训练和推理文件统一使用 UTF-8 编码的 JSONL 格式：每个非空行是一个独立 JSON 对象。
图片采用 Hugging Face 多模态内容块格式，图片块和文本块在数组中的先后顺序会被保留。

## SFT 数据

```json
{"loss_type":"sft","query":[{"type":"image","image":"/jindofs_temp/data/a.jpg"},{"type":"text","text":"图中的时间是多少？"}],"response":"12:05"}
```

字段含义：

- `loss_type`：固定为 `sft`。
- `query`：用户输入，可以包含一张或多张图片以及文本。
- `response`：模型需要学习的标准答案。

纯 SFT 训练应设置 `SFT_LOSS_WEIGHT=1`、`DPO_LOSS_WEIGHT=0` 和
`DDPO_LOSS_WEIGHT=0`。

## 标准 DPO 或 SimPO 数据

```json
{"loss_type":"dpo","query":[{"type":"image","image":"/jindofs_temp/data/a.jpg"},{"type":"text","text":"图中的时间是多少？"}],"chosen":"12:05","rejected":"12:00"}
```

字段含义：

- `query`：图片和问题组成的输入。
- `chosen`：期望模型优先生成的正确回答。
- `rejected`：需要压低概率的错误回答。

设置 `DPO_LOSS_TYPE=dpo` 时使用带参考模型的 DPO；设置
`DPO_LOSS_TYPE=simpo` 时使用不依赖参考模型的 SimPO。可选的
`focus_start` 和 `focus_end` 元数据可以把偏好信号限制在回答的指定片段。
详细格式和 Loss 定义见
[VC-Star Focus-SimPO 与 Cycle 训练说明](vcstar_focus_simpo_cycle.md)。

DPO 样本还可以额外包含 `pair_sft_query` 和 `pair_sft_response`，用于在同一条样本上
增加监督学习约束；对应权重由 `DPO_PAIR_SFT_LOSS_WEIGHT` 控制。

## 联合 DDPO 数据

显式 DDPO 格式使用一个联合问题和四种组合回答：

```json
{"loss_type":"ddpo","query":[{"type":"image","image":"/jindofs_temp/data/a.jpg"},{"type":"image","image":"/jindofs_temp/data/b.jpg"},{"type":"text","text":"分别回答第一张图和第二张图中的时间。"}],"cc":"第一张：12:05\n第二张：12:00","ci":"第一张：12:05\n第二张：12:05","ic":"第一张：12:00\n第二张：12:00","ii":"第一张：12:00\n第二张：12:05"}
```

四种回答的含义：

| 字段 | 第一张图回答 | 第二张图回答 | 偏好级别 |
|---|---|---|---|
| `cc` | 正确 | 正确 | 最优 |
| `ci` | 正确 | 错误 | 中间 |
| `ic` | 错误 | 正确 | 中间 |
| `ii` | 错误 | 错误 | 最差 |

默认偏好图包含 `cc > ci`、`cc > ic`、`ci > ii`、`ic > ii` 和 `cc > ii`。
五条边的权重分别由 `DDPO_CC_CI_WEIGHT`、`DDPO_CC_IC_WEIGHT`、
`DDPO_CI_II_WEIGHT`、`DDPO_IC_II_WEIGHT` 和 `DDPO_CC_II_WEIGHT` 控制。

如果四种回答需要使用不同图片或不同问题，可以用 `choice_queries` 替换统一的 `query`，
并分别为 `cc`、`ci`、`ic`、`ii` 提供输入。Dataset 仍兼容旧版的
`query_1`、`query_2` 和 `response_*` 字段。

## 推理数据

通用推理入口支持以下几类输入结构：

```json
{"id":"1","image":"/jindofs_temp/data/a.jpg","question":"图中的时间是多少？","answer":"12:05"}
{"id":"2","images":["/jindofs_temp/data/a.jpg","/jindofs_temp/data/b.jpg"],"prompt":"对比两张图中的时间。"}
{"id":"3","query":[{"type":"image","image":"/jindofs_temp/data/a.jpg"},{"type":"text","text":"图中的时间是多少？"}]}
{"id":"4","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"/jindofs_temp/data/a.jpg"}},{"type":"text","text":"读取钟表时间。"}]}]}
```

支持的入口字段包括：

- `question` 或 `prompt` 配合顶层 `image`/`images`。
- `query` 字符串或多模态内容块数组。
- `messages` 对话格式。
- `conversations` 旧版对话格式。

最终的 `predictions.jsonl` 会保留输入样本的全部字段，并新增：

- `_line_index`：原始文件中从 0 开始的行号。
- `prediction`：模型解码后的预测结果。
- `normalized_query`：实际送入 Processor 的标准化图片和文本内容块。

每个推理 Worker 会先写独立的 `part-XX.jsonl`，最后按原始行号合并为
`predictions.jsonl`，因此任务中断后可以继续执行。复用输出目录时，必须保证输入文件和
Worker 数量与上一次完全相同。
