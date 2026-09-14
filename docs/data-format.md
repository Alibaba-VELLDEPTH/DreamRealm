# DreamRealm family JSONL 格式

训练文件使用 UTF-8 JSONL：每个非空行是一整个 family，不允许把同一 family 的图片成员
拆散采样。图片字段可以是绝对路径，也可以是相对 `--media-root` 的路径。

## Edited family

原图与每张编辑图各自对应唯一正确回答，形成完整的 `N × N` 图片—回答评分矩阵：

```json
{"family_id":"clock-001","question":"图中的时间是多少？","original_image":"images/clock-original.jpg","original_answer":"12:05","edited_pairs":[{"image":"images/clock-edited-1.jpg","answer":"12:15"},{"image":"images/clock-edited-2.jpg","answer":"12:25"}],"original_negative_answers":[]}
```

## No-edit family

没有编辑图时，可以为原图提供零个或多个错误回答，形成 `1 × (M+1)` 评分矩阵：

```json
{"family_id":"sign-001","question":"标牌上的数字是什么？","original_image":"images/sign.jpg","original_answer":"42","edited_pairs":[],"original_negative_answers":["24","43"]}
```

## 约束

- `family_id`、`question`、`original_image` 和 `original_answer` 必须是非空字符串。
- 同一 edited family 内图片与规范化后的回答都不能重复。
- edited family 不允许再带 `original_negative_answers`。
- no-edit family 的错误回答不能重复，也不能与正确回答相同。
- 训练会对完整回答做 teacher forcing；超过 `--max-length` 的样本会报错，不会截断。
