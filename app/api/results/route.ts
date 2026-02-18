import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'

export async function GET() {
  try {
    const filePath = join(process.cwd(), 'public', 'backtest_results.json')
    const raw = await readFile(filePath, 'utf-8')
    const data = JSON.parse(raw)
    return NextResponse.json(data)
  } catch {
    return NextResponse.json(
      { error: 'No backtest results found. Run: python scripts/train_all.py' },
      { status: 404 },
    )
  }
}
