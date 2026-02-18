import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'

export async function GET() {
  try {
    const filePath = join(process.cwd(), 'public', 'sl_sweep_results.json')
    const raw = await readFile(filePath, 'utf-8')
    const data = JSON.parse(raw)
    return NextResponse.json(data)
  } catch {
    return NextResponse.json(
      { error: 'No sweep results found. Run: python scripts/sl_sweep.py' },
      { status: 404 },
    )
  }
}
