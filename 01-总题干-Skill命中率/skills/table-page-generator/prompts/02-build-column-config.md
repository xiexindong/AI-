# Step 2 · 生成配置

目标:把 Step 1 的拆解结果,填成一份**通过 schema 校验**的 `{页面名}.config.ts`。

## 前置阅读

1. `schemas/column.schema.json`(字段必须合法)
2. `templates/config.example.ts`(参考完整格式)

## 字段 → 配置的映射表(核心)

| 用户说法 | 配置怎么写 |
|---|---|
| 展示文本 | `{ field, label, render: 'text' }` |
| "按名字搜" | `filter: { type: 'input' }` |
| "按下拉选" | `filter: { type: 'select', options / optionsApi }` |
| "多选状态" | `filter: { type: 'multiSelect', optionsApi }` |
| "按时间/日期筛" | `filter: { type: 'dateRange' }, render: 'date'` |
| "选部门/组织" | `filter: { type: 'cascader', optionsApi }` |
| "数字区间" | `filter: { type: 'numberRange' }` |
| "金额/排序" | `sortable: true, render: 'money'` |
| "看合计" | `summary: 'sum'` |
| "状态用标签显示" | `render: 'tag'` |
| "点订单号跳详情" | `render: 'link'` |

## 配置模板

```ts
export const config = {
  title: '<title>',
  api: '<api>',
  columns: [ /* 每列一条 */ ],
  extra: { /* 可选 */ },
}
```

## 校验要求

生成后必须跑:

```bash
node scripts/validate-config.ts <页面名>.config.ts
```

校验失败 → 按错误信息改到通过为止,不许跳过。

## 反例(禁止)

- ❌ 给 `filter.type` 填 schema 里没有的值(如 `filter: { type: 'date' }`);
- ❌ 不写 `field`/`label`(schema 必填);
- ❌ 把筛选组件名写进配置(如 `filter: { type: 'DatePicker' }`)——筛选类型只填语义值。
