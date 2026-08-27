// templates/useTableQuery.ts —— 列表请求 hook(所有表格页共用)
// 职责:参数组装 / 请求 / loading / 错误 / 缓存 / 重试 / 联动刷新
import { useCallback, useEffect, useRef, useState } from 'react'

interface QueryOptions {
  api: string
  params: Record<string, unknown>
  refetchOn?: string[]     // 联动③:这些字段变化时自动刷新
  cacheMs?: number         // 简单内存缓存
}

interface PageResult<T> {
  list: T[]
  total: number
  summary?: Record<string, number>
}

export function useTableQuery<T = Record<string, unknown>>({ api, params, refetchOn = [], cacheMs = 30000 }: QueryOptions) {
  const [data, setData] = useState<PageResult<T> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cache = useRef(new Map<string, { t: number; d: PageResult<T> }>())
  const { page, pageSize, sort, filters } = params as any

  // 参数序列化 → 缓存 key(排序字段顺序稳定,防缓存失效)
  const cacheKey = JSON.stringify({ api, page, pageSize, sort, filters })

  const fetchData = useCallback(async (showLoading = true) => {
    showLoading && setLoading(true)
    try {
      const hit = cache.current.get(cacheKey)
      if (hit && Date.now() - hit.t < cacheMs) {
        setData(hit.d)
        return
      }
      const qs = new URLSearchParams({
        page: String(page),
        pageSize: String(pageSize),
        sortField: sort?.field ?? '',
        sortOrder: sort?.order ?? '',
        filters: JSON.stringify(filters),
      })
      const res = await fetch(`${api}?${qs}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json: PageResult<T> = await res.json()
      cache.current.set(cacheKey, { t: Date.now(), d: json })
      setData(json)
      setError(null)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [cacheKey, api, page, pageSize, sort, filters])

  useEffect(() => { fetchData() }, [fetchData])

  // 联动③:声明了 refetchOn 时,相关字段变化自动刷新(跳过首次)
  const firstRun = useRef(true)
  useEffect(() => {
    if (firstRun.current) { firstRun.current = false; return }
    if (refetchOn.some(k => k in filters)) fetchData(false)
  }, [filters, refetchOn, fetchData])

  return { data, loading, error, refetch: () => fetchData() }
}
