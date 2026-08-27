# Step 3 · 生成页面

目标:把配置 + 模板,变成用户能直接用的页面文件。

## 做法(两条路,优先 B)

### 路径 A:手动组装(无脚手架环境)
把 `templates/DataTable.tsx` 复制为 `pages/{页面名}/index.tsx`,再把配置 import 进去:

```tsx
import { DataTable } from '@/components/DataTable'
import { config } from './{页面名}.config'

export default () => <DataTable config={config} />
```

### 路径 B:脚手架(推荐)
```bash
node scripts/scaffold.ts --config {页面名}.config.ts --out pages/{页面名}
```

脚手架会自动产出:`index.tsx` + `index.css` + 注册路由。

## 生成后的文件清单(交付物)

```
pages/{页面名}/
├── index.tsx            # 页面入口(套模板)
├── {页面名}.config.ts   # 配置(Step 2 产物)
└── index.css            # 样式(脚手架生成)
```

## 必须检查

- [ ] 模板里的 `FilterFactory` 覆盖了配置里所有 `filter.type`;
- [ ] `column-renderer` 覆盖了所有 `render` 类型;
- [ ] 配置里有 `summary` 的列,模板的合计行逻辑生效;
- [ ] `extra.rowSelection` 开了 → 批量操作按钮带上了;
- [ ] 生成代码没有手写任何"筛选组件实例"(必须走工厂)。
