# Step 4 · 自检

目标:交付前,用用户的原话逐条验证,不放过一条需求。

## 对照表(把用户需求抄进来,逐条打勾)

| 用户原话 | 对应实现位置 | 检查结果 |
|---|---|---|
| "按创建时间范围筛选" | 列 createdAt 的 filter.type = 'dateRange' | ☐ |
| "状态多选" | 列 status 的 filter.type = 'multiSelect' + optionsApi | ☐ |
| "部门级联" | 列 dept 的 filter.type = 'cascader' | ☐ |
| "金额排序+合计" | 列 amount 的 sortable + summary = 'sum' | ☐ |
| "批量发货" | extra.batchActions = ['发货'] | ☐ |

## 自检脚本

```bash
node scripts/validate-config.ts {页面名}.config.ts   # 1. 配置合法性
```

再目测生成的页面:每个筛选器出现在筛选区、每列渲染正确、合计行出现。

## 常见遗漏(重点查)

1. **筛选后页码没重置**(模板已处理,确认没被改动);
2. 多选筛选项是**接口来的**却忘了配 `optionsApi`;
3. 金额列没格式化(render 忘了写 'money');
4. 批量操作选了行才可点(disabled 逻辑);
5. 空状态文案。

## 通过标准

用户提的每条需求都能在页面里**指出来**,且配置通过了校验 → 通过。
