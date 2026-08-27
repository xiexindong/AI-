import React from 'react'
import { Tag } from 'antd'
import DataTable, { ColumnConfig, ListParams, PageResult } from './DataTable'

/* ==================== 示例:用户管理页(新页面唯一要写的东西) ==================== */

interface User {
  id: number
  name: string
  age: number
  status: 1 | 2
  createdAt: string
}

/* ① 接口函数:按约定接收 分页 + 排序 + 筛选 参数 */
async function fetchUsers(params: ListParams): Promise<PageResult<User>> {
  const qs = new URLSearchParams(params as any).toString()
  const res = await fetch(`/api/users?${qs}`)
  return res.json()
}

/* ② 动态筛选项:状态选项从接口来(也可以写死) */
const fetchStatusOptions = () =>
  Promise.resolve([
    { label: '启用', value: 1 },
    { label: '禁用', value: 2 },
  ])

/* ③ 列配置:字段不同,只改这里 */
const columns: ColumnConfig<User>[] = [
  { field: 'id', label: 'ID', width: 80, sortable: true },
  { field: 'name', label: '姓名', filter: { type: 'input' } },
  { field: 'age', label: '年龄', width: 100, sortable: true, filter: { type: 'number' } },
  {
    field: 'status',
    label: '状态',
    width: 100,
    filter: { type: 'select', optionsApi: fetchStatusOptions },
    render: (v) => <Tag color={v === 1 ? 'green' : 'red'}>{v === 1 ? '启用' : '禁用'}</Tag>,
  },
  {
    field: 'createdAt',
    label: '创建时间',
    width: 180,
    sortable: true,
    filter: { type: 'dateRange' },
  },
]

/* ④ 页面入口:一行组装 */
export default function UserListPage() {
  return <DataTable columns={columns} api={fetchUsers} rowKey="id" />
}
