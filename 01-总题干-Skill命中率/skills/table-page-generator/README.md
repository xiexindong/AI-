# table-page-generator 大型 Skill 组织地图

> 这是一个**真实可用的完整 Skill**,专门用来演示"大型 Skill 的组织结构是怎么划分的"。
> 每个文件都是真内容,不是占位符。

## 一、目录总览(组织划分规律一眼看懂)

```
table-page-generator/
├── SKILL.md                     # ① 唯一入口:LLM 加载它,它再去指挥其他文件
├── manifest.json                # ② 元信息:版本/权限/依赖(注册表登记用)
├── README.md                    # ③ 组织地图(本文件,给人看的,LLM 不加载)
│
├── prompts/                     # ④ 子指令:大流程拆成 5 个小步骤文件
│   ├── 01-analyze-requirements.md
│   ├── 02-build-column-config.md
│   ├── 03-generate-page.md
│   ├── 04-self-review.md
│   └── 05-edge-cases.md
│
├── templates/                   # ⑤ 代码模板:生成产物的骨架(写死)
│   ├── DataTable.tsx            #    通用表格壳(分页/排序/筛选/合计)
│   ├── useTableQuery.ts         #    请求 hook(参数组装/缓存/重试)
│   └── config.example.ts        #    配置示例(LLM 产出配置的参考格式)
│
├── fragments/                   # ⑥ 复用片段:确定性映射逻辑(写死)
│   ├── filter-factory.tsx       #    filter.type → 筛选组件
│   └── column-renderer.tsx      #    render 类型 → 单元格渲染
│
├── schemas/                     # ⑦ 参数契约:校验 LLM 生成的配置
│   └── column.schema.json
│
├── scripts/                     # ⑧ 可执行工具:跑得动的 Node 脚本
│   ├── scaffold.ts              #    按配置生成页面文件的脚手架
│   └── validate-config.ts       #    用 schema 校验配置的脚本
│
└── references/                  # ⑨ 参考资料:锦上添花的规范文档
    └── best-practices.md
```

## 二、划分规律:按"谁在变"分目录

| 目录 | 存放什么 | 变化频率 | 谁在用 |
|---|---|---|---|
| `prompts/` | LLM 的决策流程(怎么想) | 高(规则常迭代) | LLM |
| `templates/` | 生成产物的骨架(怎么搭) | 低(稳定) | 代码生成 |
| `fragments/` | 配置→组件的映射(怎么选) | 低(确定性) | 代码 |
| `schemas/` | 参数契约(怎么校验) | 中(跟着配置变) | 校验器 |
| `scripts/` | 可执行工具(怎么落地) | 中 | 运行时 |
| `references/` | 规范参考(怎么更好) | 低 | LLM 按需查 |

**一句话:** 需要 LLM"想"的放 prompts,不需要 LLM"想"的写死成代码(templates/fragments/scripts),要别人看的放 references。

## 三、为什么这样拆?(大型 Skill 的三大原则)

1. **入口唯一**:LLM 只加载 SKILL.md,SKILL.md 里用"先读 X 再读 Y"指挥子指令 → 上下文按需加载,不一次性全塞;
2. **流程可维护**:加一个筛选类型 → 改 fragments + schema,不用动 SKILL.md;
3. **人机分离**:给人看的(README/references)和给 LLM 看的(prompts)分开,互不污染。
