// templates/config.example.ts —— 配置参考格式(LLM 生成配置时的标准样板)
// 用法:复制改字段,务必通过 scripts/validate-config.ts 校验
import type { PageConfig } from '../schemas/column.schema.json'

export const config: PageConfig = {
  title: '订单管理',
  api: '/orders',

  columns: [
    // 文本筛选 + 链接渲染
    { field: 'orderNo', label: '订单号', width: 180,
      render: 'link', filter: { type: 'input' } },

    // 日期范围筛选 + 排序
    { field: 'createdAt', label: '创建时间', width: 170, sortable: true,
      render: 'date', filter: { type: 'dateRange' } },

    // 多选下拉:选项来自接口
    { field: 'status', label: '状态', width: 100,
      render: 'tag', filter: { type: 'multiSelect', optionsApi: '/orders/status-options' } },

    // 级联选择:部门树
    { field: 'dept', label: '所属部门', width: 140,
      render: 'text', filter: { type: 'cascader', optionsApi: '/depts/tree' } },

    // 金额:排序 + 合计行
    { field: 'amount', label: '金额', width: 130, sortable: true,
      render: 'money', summary: 'sum' },

    // 数字区间筛选
    { field: 'quantity', label: '数量', width: 100,
      render: 'number', filter: { type: 'numberRange' } },
  ],

  extra: {
    rowSelection: true,
    batchActions: ['发货', '关闭'],
    defaultSort: { field: 'createdAt', order: 'desc' },
    refetchOn: ['status'],          // 状态变化 → 自动刷新
  },
}
