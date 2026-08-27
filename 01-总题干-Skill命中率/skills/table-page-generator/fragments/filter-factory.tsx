// fragments/filter-factory.tsx —— 筛选类型 → 组件 的确定性映射
// 新增筛选类型:① 在这里加 case ② 在 schemas/column.schema.json 加 enum 值
// 永远不要在页面/模板里手写筛选组件实例
import { Input, Select, DatePicker, Cascader, InputNumber, Space } from 'antd'
import type { ColumnConfig } from '../schemas/column.schema.json'

interface Props {
  column: ColumnConfig
  value: unknown
  onChange: (v: unknown) => void
}

const loadOptions = async (api?: string): Promise<{ label: string; value: unknown }[]> => {
  if (!api) return []
  const res = await fetch(api)
  return res.json()
}

export function FilterFactory({ column, value, onChange }: Props) {
  const { filter } = column
  const load = () => loadOptions(filter.optionsApi)   // 需要时再拉选项

  switch (filter.type) {
    case 'input':
      return (
        <Input
          placeholder={`按${column.label}`}
          value={value as string}
          onChange={e => onChange(e.target.value)}
          allowClear
        />
      )

    case 'select':
      return <Select value={value} onChange={onChange} options={filter.options} onDropdownOpen={load} allowClear style={{ minWidth: 120 }} />

    case 'multiSelect':
      return (
        <Select
          mode="multiple"
          placeholder={`多选${column.label}`}
          value={value as string[]}
          onChange={onChange}
          options={filter.options}
          onDropdownOpen={load}
          allowClear
          style={{ minWidth: 160 }}
        />
      )

    case 'dateRange':
      return <DatePicker.RangePicker value={value as any} onChange={onChange} />

    case 'cascader':
      return <Cascader value={value as any} onChange={onChange} options={filter.options as any} onDropdownOpen={load} placeholder={`选择${column.label}`} />

    case 'numberRange':
      return (
        <Space.Compact>
          <InputNumber placeholder="最小" value={(value as any)?.[0]} onChange={v => onChange([v, (value as any)?.[1]])} />
          <InputNumber placeholder="最大" value={(value as any)?.[1]} onChange={v => onChange([(value as any)?.[0], v])} />
        </Space.Compact>
      )

    default:
      return null
  }
}
