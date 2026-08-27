// templates/DataTable.tsx —— 通用表格壳(所有生成的页面共用,禁止手改)
// 职责:分页 / 排序 / 筛选 / 请求 / loading / 错误态 / 空态 / 合计行 / 批量操作
import { useMemo, useState } from 'react'
import { Table, Space } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useTableQuery } from './useTableQuery'
import { FilterFactory } from '../fragments/filter-factory'
import { ColumnRenderer } from '../fragments/column-renderer'
import type { PageConfig } from '../schemas/column.schema.json'

interface Props {
  config: PageConfig
}

export function DataTable({ config }: Props) {
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [sort, setSort] = useState<{ field?: string; order?: 'asc' | 'desc' }>(
    config.extra?.defaultSort ?? {},
  )
  const [filters, setFilters] = useState<Record<string, unknown>>({})

  // 联动①:筛选变化 → 重置页码
  const onFilterChange = (key: string, value: unknown) => {
    setFilters(prev => ({ ...prev, [key]: value }))
    setPage(1)
  }

  const { data, loading, error, refetch } = useTableQuery({
    api: config.api,
    params: { page, pageSize, sort, filters },
  })

  // 联动②:summary 列聚合数据(后端算好或前端算)
  const summaryData = useMemo(() => data?.summary ?? {}, [data])

  const columns: ColumnsType = [
    ...(config.extra?.rowSelection ? [{
      title: <input type="checkbox" />,
      key: '__selection',
      width: 40,
    }] : []),
    ...config.columns.map(col => ({
      title: (
        <Space size={4}>
          {col.label}
          {col.sortable && (
            <button onClick={() => setSort(prev => ({
              field: col.field,
              order: prev.field === col.field && prev.order === 'asc' ? 'desc' : 'asc',
            }))}>↕</button>
          )}
        </Space>
      ),
      dataIndex: col.field,
      key: col.field,
      width: col.width,
      sorter: col.sortable,
      render: (_: unknown, row: Record<string, unknown>) => (
        <ColumnRenderer column={col} row={row} />
      ),
    })),
  ]

  return (
    <div className="data-table" data-testid="data-table">
      {/* 筛选区:全部走 FilterFactory,不在这里手写任何筛选组件 */}
      <div className="filters" style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        {config.columns
          .filter(c => c.filter)
          .map(col => (
            <FilterFactory
              key={col.field}
              column={col}
              value={filters[col.field]}
              onChange={v => onFilterChange(col.field, v)}
            />
          ))}
        <button onClick={refetch}>查询</button>
        {config.extra?.batchActions && (
          <Space>
            {config.extra.batchActions.map((action: string) => (
              <button key={action} disabled={!selectedCount}>
                {action}
              </button>
            ))}
          </Space>
        )}
      </div>

      <Table
        columns={columns}
        dataSource={data?.list ?? []}
        rowKey="id"
        loading={loading}
        error={error}
        pagination={{
          current: page,
          pageSize,
          total: data?.total ?? 0,
          onChange: (p, s) => { setPage(p); setPageSize(s) },
          showTotal: t => `共 ${t} 条`,
        }}
        summary={() => config.columns.some(c => c.summary) ? (
          <Table.Summary.Row>
            {config.columns.map(col =>
              col.summary === 'sum' ? (
                <Table.Summary.Cell key={col.field}>
                  合计:{summaryData[col.field] ?? 0}
                </Table.Summary.Cell>
              ) : (
                <Table.Summary.Cell key={col.field} />
              ),
            )}
          </Table.Summary.Row>
        ) : undefined}
      />
    </div>
  )
}
