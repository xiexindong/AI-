// fragments/column-renderer.tsx —— 渲染类型 → 单元格组件 的确定性映射
// 新增渲染类型:① 这里加 case ② schema 的 enum 加值
import { Tag, Typography } from 'antd'
import type { ColumnConfig } from '../schemas/column.schema.json'

interface Props {
  column: ColumnConfig
  row: Record<string, unknown>
}

const STATUS_COLORS: Record<string, string> = {
  pending: 'orange',     // 待处理
  paid: 'blue',          // 已支付
  shipped: 'geekblue',   // 已发货
  done: 'green',         // 已完成
  closed: 'red',         // 已关闭
}

export function ColumnRenderer({ column, row }: Props) {
  const raw = row[column.field]
  if (raw == null || raw === '') return <span style={{ color: '#bbb' }}>-</span>

  switch (column.render) {
    case 'text':
      return <span>{String(raw)}</span>

    case 'date':
      return <span>{new Date(raw as string).toLocaleString('zh-CN')}</span>

    case 'money':
      return <span>{Number(raw).toLocaleString('zh-CN', { minimumFractionDigits: 2 })}</span>

    case 'number':
      return <span>{Number(raw).toLocaleString('zh-CN')}</span>

    case 'tag':
      return <Tag color={STATUS_COLORS[String(raw)] ?? 'default'}>{String(raw)}</Tag>

    case 'link':
      return <Typography.Link>{String(raw)}</Typography.Link>

    case 'custom':
      // 配置里给 column.renderCustom 传一个 React 组件即可扩展
      return column.renderCustom ? column.renderCustom(raw, row) : <span>{String(raw)}</span>

    default:
      return <span>{String(raw)}</span>
  }
}
