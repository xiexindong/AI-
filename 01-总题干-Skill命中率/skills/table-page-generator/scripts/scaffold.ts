// scripts/scaffold.ts —— 脚手架:按配置生成页面文件
// 用法:node scripts/scaffold.ts --config order-page.config.ts --out pages/order-page
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

// 模板路径:相对本文件往上一级进 templates/
const TEMPLATE_DIR = resolve(dirname(new URL(import.meta.url).pathname), '../templates')

function parseArgs(argv: string[]) {
  const args: Record<string, string> = {}
  for (let i = 0; i < argv.length; i += 2) {
    args[argv[i].replace(/^--/, '')] = argv[i + 1]
  }
  return args
}

function main() {
  const { config, out } = parseArgs(process.argv.slice(2))
  if (!config || !out) {
    console.error('用法:node scaffold.ts --config <配置.ts> --out <输出目录>')
    process.exit(1)
  }

  // 1. 读配置(真实场景里转译 ts 后 import;这里做轻量占位)
  const configContent = readFileSync(config, 'utf-8')
  const titleMatch = configContent.match(/title:\s*'([^']+)'/)
  const title = titleMatch?.[1] ?? 'Untitled'

  // 2. 读模板
  const dataTable = readFileSync(join(TEMPLATE_DIR, 'DataTable.tsx'), 'utf-8')

  // 3. 组装页面入口
  const pageEntry = `// 由 table-page-generator 脚手架生成,勿手改模板部分
import { DataTable } from '@/components/DataTable'
import { config } from './${config}'

export default function Page() {
  return <DataTable config={config} />
}
`

  // 4. 落盘
  mkdirSync(out, { recursive: true })
  writeFileSync(join(out, 'index.tsx'), pageEntry)
  writeFileSync(join(out, config), configContent)
  writeFileSync(join(out, 'index.css'), `.data-table { padding: 16px; }`)

  console.log(`✅ 已生成页面:${out} (${title})`)
  console.log(`   - index.tsx       页面入口(套模板)`)
  console.log(`   - ${config}   配置`)
  console.log(`   - index.css        样式`)
}

main()
