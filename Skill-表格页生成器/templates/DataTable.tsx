import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { DatePicker, Input, Pagination, Select, Space, Table } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import type { Dayjs } from 'dayjs'

/* ==================== 一、列配置 Schema(每个页面的"说明书") ==================== */

export interface FilterOption {
  label: string
  value: string | number
}

export interface ColumnFilterConfig {
  type: 'input' | 'number' | 'select' | 'dateRange'
  options?: FilterOption[]                    // select 的静态选项
  optionsApi?: () => Promise<FilterOption[]>  // select 选项来自接口(动态)
  placeholder?: string
}

export interface ColumnConfig<T = any> {
  field: string                                // 字段名,对应接口返回的 key
  label: string                                // 表头文字
  width?: number
  sortable?: boolean                           // 是否可排序(点击表头)
  hidden?: boolean                             // 渲染但不显示(预留列)
  filter?: ColumnFilterConfig                  // 可筛选 + 怎么筛
  render?: (value: any, row: T) => React.ReactNode  // 自定义单元格
}

/* ==================== 二、请求参数/响应约定(前后端对齐) ==================== */

export interface ListParams {
  page: number
  pageSize: number
  sortField?: string
  sortOrder?: 'ascend' | 'descend'
  [key: string]: any                           // 筛选条件平铺
}

export interface PageResult<T> {
  list: T[]
  total: number
}

/* ==================== 三、单个筛选控件(动态选项也在这管) ==================== */

function FilterControl<T>({ col, value, onChange }: {
  col: ColumnConfig<T>
  value: any
  onChange: (v: any) => void
}) {
  const f = col.filter!
  const [opts, setOpts] = useState<FilterOption[]>(f.options ?? [])

  // 选项来自接口时,挂载后拉一次
  useEffect(() => {
    if (f.optionsApi) f.optionsApi().then(setOpts)
  }, [f.optionsApi])

  const common = { allowClear: true, placeholder: f.placeholder ?? `请输入${col.label}` }

  switch (f.type) {
    case 'input':
      return <Input {...common} value={value} onChange={(e) => onChange(e.target.value)} />
    case 'number':
      return <Input type="number" {...common} value={value} onChange={(e) => onChange(e.target.value)} />
    case 'select':
      return <Select {...common} style={{ minWidth: 120 }} value={value} options={opts} onChange={onChange} />
    case 'dateRange':
      return (
        <DatePicker.RangePicker
          {...common}
          value={value}
          onChange={(dates) =>
            onChange(
              dates ? [dates[0]?.format('YYYY-MM-DD'), dates[1]?.format('YYYY-MM-DD')] : undefined,
            )
          }
        />
      )
  }
}

/* ==================== 四、通用表格组件(写一次,所有页面复用) ==================== */

interface DataTableProps<T = any> {
  columns: ColumnConfig<T>[]
  api: (params: ListParams) => Promise<PageResult<T>>
  rowKey?: string
  extraParams?: Record<string, any>            // 固定附加参数(如固定 orgId)
  defaultPageSize?: number
}

export default function DataTable<T extends Record<string, any>>({
  columns,
  api,
  rowKey = 'id',
  extraParams,
  defaultPageSize = 20,
}: DataTableProps<T>) {
  const [list, setList] = useState<T[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(defaultPageSize)
  const [filters, setFilters] = useState<Record<string, any>>({})
  const [sort, setSort] = useState<{ field: string; order: 'asc' | 'desc' }>()

  // 筛选条件变化 → 强制回第 1 页,避免停在空页
  const changeFilter = (key: string, value: any) => {
    setFilters((prev) => ({ ...prev, [key]: value }))
    setPage(1)
  }

  // 自动拼参数:分页 + 排序 + 筛选 + 附加参数
  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params: ListParams = {
        page,
        pageSize,
        sortField: sort?.field,
        sortOrder: sort?.order,
        ...extraParams,
        ...filters,
      }
      const res = await api(params)
      setList(res.list)
      setTotal(res.total)
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, sort, filters, api, extraParams])

  useEffect(() => {
    load()
  }, [load])

  // 列配置 → antd columns(可排序的接通用排序逻辑)
  const antdColumns = useMemo<ColumnsType<T>>(
    () =>
      columns
        .filter((c) => !c.hidden)
        .map((c) => ({
          title: c.label,
          dataIndex: c.field,
          key: c.field,
          width: c.width,
          sorter: c.sortable ? true : false,
          sortOrder: sort?.field === c.field ? sort.order : null,
          render: c.render,
        })),
    [columns, sort],
  )

  const filterableColumns = useMemo(() => columns.filter((c) => c.filter), [columns])

  return (
    <div>
      {/* 筛选栏:遍历有 filter 配置的列,自动渲染对应控件 */}
      {filterableColumns.length > 0 && (
        <Space wrap style={{ marginBottom: 16 }}>
          {filterableColumns.map((col) => (
            <FilterControl
              key={col.field}
              col={col}
              value={filters[col.field]}
              onChange={(v) => changeFilter(col.field, v)}
            />
          ))}
        </Space>
      )}

      <Table<T>
        rowKey={rowKey}
        columns={antdColumns}
        dataSource={list}
        loading={loading}
        pagination={false}
        onChange={(p, _f, s) => {
          // 点击可排序表头 → 更新排序并重新请求
          const so = Array.isArray(s) ? s[0] : s
          if (so && so.field) {
            setSort({ field: String(so.field), order: so.order === 'ascend' ? 'asc' : 'desc' })
          } else {
            setSort(undefined)
          }
        }}
      />

      {/* 分页 */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
        <Pagination
          current={page}
          pageSize={pageSize}
          total={total}
          showSizeChanger
          showTotal={(t) => `共 ${t} 条`}
          onChange={(p) => setPage(p)}
          onShowSizeChange={(_p, size) => {
            setPageSize(size)
            setPage(1)
          }}
        />
      </div>
    </div>
  )
}
