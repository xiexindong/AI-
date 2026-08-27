---
name: table-page-generator
description: >
  当用户想要"后台数据管理页、列表页、表格页"时使用。典型需求:分页列表、
  按条件筛选(文本/下拉/日期范围/级联/数字区间)、列排序、合计行、批量操作。
  触发词:表格、列表、数据管理、分页、筛选、排序、合计、批量、CRUD 页面。
negative_description: >
  不要用于:表单页(新增/编辑/详情)、图表可视化、流程审批、纯前端组件改造。
  如果用户只是问"表格怎么写"而不是要生成页面,也不要使用。
version: 2.1.0
entry: SKILL.md
parameters:
  type: object
  required: [title, api, columns]
  properties:
    title:
      type: string
      description: 页面标题,如"订单管理"
    api:
      type: string
      description: 列表接口路径,如 GET /orders
    columns:
      type: array
      description: 列配置,详见 schemas/column.schema.json
    extra:
      type: object
      description: 额外能力:rowSelection/batchActions/defaultSort 等
permissions:
  - files:read
  - files:write
  - terminal:read
---

# 表格页生成器

本 Skill 用"配置驱动"的方式生成后台表格页面。**你只需要产出配置文件,页面代码由模板和脚手架生成。**

## 执行流程(严格按序,每步先读对应子指令)

### Step 1 · 分析需求
先读 `prompts/01-analyze-requirements.md`,把用户的话拆成 title/api/columns/extra 四部分。

### Step 2 · 生成配置
先读 `prompts/02-build-column-config.md`,再读 `schemas/column.schema.json` 和 `templates/config.example.ts`,产出 `{页面名}.config.ts`。

### Step 3 · 生成页面
先读 `prompts/03-generate-page.md`,调用 `scripts/scaffold.ts` 按配置生成页面文件。

### Step 4 · 自检
先读 `prompts/04-self-review.md`,逐条对照用户需求检查遗漏。

### Step 5 · 边界情况
遇到用户没说清、或配置表达不了的场景,先读 `prompts/05-edge-cases.md` 再决定。

## 硬性规则(违反即重做)

1. 所有"筛选类型→组件"的映射必须在 `fragments/filter-factory.tsx` 里,不要在页面里手写;
2. 生成的文件必须通过 `scripts/validate-config.ts` 校验;
3. 用户提的每个筛选/排序/合计需求,都要能在最终页面找到对应物;
4. 交付时告诉用户"改页面 = 改配置",不要让他改模板。
