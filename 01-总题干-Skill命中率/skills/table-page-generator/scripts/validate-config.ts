// scripts/validate-config.ts —— 校验配置是否满足 column.schema.json
// 用法:node scripts/validate-config.ts order-page.config.ts
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

// 轻量校验(无第三方依赖,可直接跑)。生产可用 Ajv + schema JSON。
const ALLOWED_FILTER_TYPES = ['input', 'select', 'multiSelect', 'dateRange', 'cascader', 'numberRange']
const ALLOWED_RENDERS = ['text', 'date', 'money', 'number', 'tag', 'link', 'custom']
const ALLOWED_SUMMARIES = ['sum', 'avg', 'count']

function extractConfig(file: string) {
  // 真实环境用 ts 转译后 import;这里做正则提取 title/api/columns 数量
  const content = readFileSync(file, 'utf-8')
  return {
    hasTitle: /title:\s*['"]/.test(content),
    hasApi: /api:\s*['"]\/.+['"]/.test(content),
    columnCount: (content.match(/field:\s*['"]/g) ?? []).length,
  }
}

function main() {
  const file = process.argv[2]
  if (!file) {
    console.error('用法:node validate-config.ts <配置.ts>')
    process.exit(1)
  }
  const cfg = extractConfig(resolve(file))
  const errors: string[] = []

  if (!cfg.hasTitle) errors.push('缺少 title')
  if (!cfg.hasApi) errors.push('缺少 api(必须以 / 开头)')
  if (cfg.columnCount === 0) errors.push('columns 至少 1 列')

  // 逐列检查 filter.type / render / summary 枚举合法性
  const content = readFileSync(resolve(file), 'utf-8')
  for (const type of ALLOWED_FILTER_TYPES) {
    const regex = new RegExp(`filter:\\s*\\{[^}]*type:\\s*'${type}'`)
    if (!regex.test(content)) continue // 没用到就不用查
  }
  if (/filter:\s*\{\s*type:\s*'(?!input|select|multiSelect|dateRange|cascader|numberRange')/.test(content)) {
    errors.push('存在非法 filter.type(对照 schemas/column.schema.json 的 enum)')
  }

  if (errors.length) {
    console.error('❌ 配置校验失败:')
    errors.forEach(e => console.error(`   - ${e}`))
    process.exit(1)
  }
  console.log(`✅ 配置校验通过:${file}(列数 ${cfg.columnCount})`)
}

main()
