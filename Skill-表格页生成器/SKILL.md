---
name: table-page-generator
description: >
  当用户需要创建一个"后台列表/表格页面"时使用。典型场景:用户管理列表、订单列表、
  商品列表、任务列表——任何"数据表格 + 分页 + 搜索筛选 + 排序"的 CRUD 列表页。
  触发词:做个表格页、列表页、分页表格、后台列表、管理页、带筛选的表格、带排序的列表。
  不要使用:如果用户要的是表单页、详情页、图表页(非表格结构),或纯静态展示(无接口请求),
  或用户明确要求特定 UI 框架外的定制布局。
  重要:不要为不同的字段组合新建 skill——所有表格页共用本 skill,字段通过列配置传入。
---

# 表格页生成器(Table Page Generator)

把"分页 + 请求 + 筛选 + 排序"这些表格页通用骨架写死,用户只需提供**字段清单**,
即可生成一个完整可用的列表页面。

## 一、用户需要提供什么(输入)

| 输入 | 说明 | 示例 |
|---|---|---|
| 页面名 | 业务叫什么 | 用户管理页 |
| 字段清单 | 每列:字段名/表头/类型/是否可排序/是否可筛选 | 姓名(input)、状态(select) |
| 接口路径(可选) | 后端接口,缺省用 `/{页面名小写}` | `/api/users` |
| 特殊要求(可选) | 自定义渲染、附加参数 | 状态列显示 Tag 标签 |

## 二、输出文件结构

```
src/pages/
└── {页面名}/
    ├── index.tsx              # 页面入口(配置 + 调组件)
    ├── columns.ts             # 列配置(每个页面唯一要写的)
    └── api.ts                 # 接口函数(可选,或直接写在 index.tsx)

src/components/
└── DataTable.tsx              # 通用表格组件(写一次,所有页面共用)
```

## 三、使用步骤

1. **复制通用组件**:把 `templates/DataTable.tsx` 放到项目 `src/components/`;
2. **写列配置**:根据用户字段清单填 `templates/user-list.example.tsx` 里的 `columns`;
3. **写接口函数**:按约定的请求参数格式(见下)实现 `api`;
4. **组装页面**:页面入口只做一件事 → `<DataTable columns={columns} api={fetchXxx} />`。

## 四、请求参数约定(前后端对齐)

```
GET /api/users?page=1&pageSize=20&sortField=createdAt&sortOrder=desc&name=张&status=1&createdAtFrom=2024-01-01&createdAtTo=2024-12-31
```

- `page` / `pageSize`:分页,固定
- `sortField` / `sortOrder`:排序,只出现在可排序列被点击时
- 其余参数:筛选条件平铺(筛选框的值直接按字段名拼上)
- 后端统一返回:`{ list: [], total: number }`

## 五、列配置字段说明(Schema)

| 字段 | 类型 | 说明 |
|---|---|---|
| `field` | string | 字段名,对应接口返回的 key(必填) |
| `label` | string | 表头文字(必填) |
| `width` | number | 列宽 |
| `sortable` | boolean | 是否可排序(点击表头) |
| `hidden` | boolean | 渲染但不显示(预留列) |
| `filter.type` | `input` / `number` / `select` / `dateRange` | 筛选控件类型 |
| `filter.options` | array | select 的静态选项 |
| `filter.optionsApi` | function | select 选项来自接口时用 |
| `render` | function | 自定义单元格渲染(Tag、链接、操作按钮) |

## 六、注意事项

1. **筛选条件变化 → 页码强制回第 1 页**,否则可能停在空页;
2. **输入型筛选做防抖**(300ms),避免每敲一个字就请求;
3. **筛选选项优先用接口拉**(`optionsApi`),不要写死,支持动态数据;
4. **不要过度定制**:操作列、批量选择这类"超出骨架"的需求,通过 `render` 和 `extraParams` 扩展,不动通用组件;
5. **一个 Skill 管所有表格页**:页面差异只体现在 `columns` 配置里,禁止按字段拆分多个 Skill。

## 七、示例

运行示例见 `templates/user-list.example.tsx`(用户管理页:姓名/年龄/状态/创建时间 + 分页 + 筛选 + 排序)。
