"use client"

import { useState, useEffect, useCallback, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from "recharts"
import { Shield, TrendingUp, Activity, Target, RefreshCw, ArrowLeft, Zap } from "lucide-react"
import Link from "next/link"

// ── types ────────────────────────────────────────────────────────

type Metrics = {
  algorithm: string
  total_return: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  total_trades: number
  avg_trade_pnl: number
  final_balance: number
  sl_hits: number
  tp_hits: number
  sl_pct?: string
  tp_pct?: string
}

type SweepResults = {
  sl_levels: string[]
  tp_levels: string[]
  no_sl_baseline: Record<string, Metrics>
  grid: Record<string, Record<string, Record<string, Metrics>>>
  best_per_algo: Record<string, Metrics>
}

const ALGO_COLORS: Record<string, string> = {
  "DQN":         "#f59e0b",
  "Double DQN":  "#3b82f6",
  "Dueling DQN": "#10b981",
  "A2C":         "#8b5cf6",
}

const METRICS = [
  { key: "sharpe_ratio",  label: "Sharpe Ratio",   fmt: (v: number) => v.toFixed(2) },
  { key: "total_return",  label: "Total Return %",  fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` },
  { key: "max_drawdown",  label: "Max Drawdown %",  fmt: (v: number) => `-${v.toFixed(2)}%` },
  { key: "win_rate",      label: "Win Rate %",      fmt: (v: number) => `${v.toFixed(1)}%` },
] as const

// ── color scales ─────────────────────────────────────────────────

function sharpeColor(val: number, min: number, max: number): string {
  if (max === min) return '#1e293b'
  const t = Math.max(0, Math.min(1, (val - min) / (max - min)))
  // Red → Yellow → Green
  if (t < 0.5) {
    const r = 220, g = Math.round(60 + t * 2 * 160), b = 60
    return `rgb(${r},${g},${b})`
  } else {
    const r = Math.round(220 - (t - 0.5) * 2 * 180), g = 200, b = 60
    return `rgb(${r},${g},${b})`
  }
}

function textColor(bg: string): string {
  // Simplified contrast check
  const m = bg.match(/rgb\((\d+),(\d+),(\d+)\)/)
  if (!m) return '#e2e8f0'
  const luma = 0.299 * +m[1] + 0.587 * +m[2] + 0.114 * +m[3]
  return luma > 140 ? '#0a0e17' : '#e2e8f0'
}


// ── component ───────────────────────────────────────────────────

export default function SweepDashboard() {
  const [data, setData] = useState<SweepResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [metric, setMetric] = useState<string>("sharpe_ratio")

  const fetchResults = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const res = await fetch("/api/sweep")
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.error || "Failed to load")
      }
      setData(await res.json())
    } catch (e: any) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetchResults() }, [fetchResults])

  const algoNames = useMemo(() => data ? Object.keys(data.grid) : [], [data])
  const tpLabel = useMemo(() => data?.tp_levels?.[0] ?? 'None', [data])
  const hasTp = useMemo(() => data?.tp_levels && data.tp_levels.length > 1, [data])

  // Build line chart data: { sl: "0.5%", DQN: 1.23, "Double DQN": 0.87, ... }
  const lineData = useMemo(() => {
    if (!data) return []
    return data.sl_levels.map(sl => {
      const point: Record<string, any> = { sl }
      for (const algo of algoNames) {
        const cell = data.grid[algo]?.[sl]?.[tpLabel]
        if (cell) point[algo] = (cell as any)[metric]
      }
      return point
    })
  }, [data, algoNames, metric, tpLabel])

  // Heatmap min/max for the selected metric
  const { heatMin, heatMax } = useMemo(() => {
    if (!data) return { heatMin: 0, heatMax: 1 }
    let vals: number[] = []
    for (const algo of algoNames) {
      for (const sl of data.sl_levels) {
        const cell = data.grid[algo]?.[sl]?.[tpLabel]
        if (cell) vals.push((cell as any)[metric])
      }
    }
    return { heatMin: Math.min(...vals), heatMax: Math.max(...vals) }
  }, [data, algoNames, metric, tpLabel])

  // Current metric config
  const metricConfig = METRICS.find(m => m.key === metric) || METRICS[0]

  // ── loading / error ─────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0e17] flex items-center justify-center">
        <div className="text-center space-y-4">
          <RefreshCw className="h-10 w-10 text-emerald-400 animate-spin mx-auto" />
          <p className="text-slate-400 text-lg">Loading sweep results…</p>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-[#0a0e17] flex items-center justify-center p-6">
        <Card className="max-w-lg w-full bg-[#111827] border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <Shield className="h-6 w-6 text-amber-400" />
              No Sweep Results
            </CardTitle>
            <CardDescription className="text-slate-400">
              Run the stop-loss sweep after training your models.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="bg-[#0a0e17] rounded-lg p-4 font-mono text-sm text-slate-300 space-y-1">
              <p className="text-slate-500"># Train models first (if not done)</p>
              <p>python scripts/train_all.py --episodes 50</p>
              <p className="text-slate-500 mt-2"># Run SL sweep (0.5% to 5%)</p>
              <p>python scripts/sl_sweep.py</p>
              <p className="text-slate-500 mt-2"># Or with TP levels too</p>
              <p>python scripts/sl_sweep.py --tp 0.01 0.02 0.03 0.04 0.05</p>
            </div>
            <div className="flex gap-2">
              <Link href="/">
                <Button variant="outline" size="sm" className="border-slate-600 text-slate-300 hover:bg-slate-700">
                  <ArrowLeft className="h-4 w-4 mr-1" /> Main Dashboard
                </Button>
              </Link>
              <Button onClick={fetchResults} variant="outline" size="sm" className="border-slate-600 text-slate-300 hover:bg-slate-700">
                <RefreshCw className="h-4 w-4 mr-1" /> Retry
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // ── render ──────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-[#0a0e17] text-slate-100 p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-amber-500/10">
              <Shield className="h-8 w-8 text-amber-400" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
                Stop-Loss Sweep Analysis
              </h1>
              <p className="text-slate-500 text-sm">
                4 algorithms × {data.sl_levels.length} SL levels{hasTp ? ` × ${data.tp_levels.length} TP levels` : ''} · SOL/USDT 15m
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Link href="/">
              <Button variant="outline" size="sm" className="border-slate-700 text-slate-400 hover:text-slate-100 hover:bg-slate-800">
                <ArrowLeft className="h-4 w-4 mr-1" /> Main
              </Button>
            </Link>
            <Button onClick={fetchResults} variant="outline" size="sm" className="border-slate-700 text-slate-400 hover:text-slate-100 hover:bg-slate-800">
              <RefreshCw className="h-4 w-4 mr-1" /> Refresh
            </Button>
          </div>
        </div>

        {/* Best per algo summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {algoNames.map(name => {
            const best = data.best_per_algo[name]
            const base = data.no_sl_baseline[name]
            const improved = best && base ? best.sharpe_ratio > base.sharpe_ratio : false
            return (
              <Card key={name} className="bg-[#111827] border-slate-800">
                <CardContent className="pt-4 pb-3 px-4">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="w-2 h-2 rounded-full" style={{ background: ALGO_COLORS[name] }} />
                    <span className="text-xs text-slate-500 uppercase tracking-wider">{name}</span>
                  </div>
                  <div className="text-lg font-bold text-slate-100">
                    SL {best?.sl_pct || '—'}
                    {best?.tp_pct && best.tp_pct !== 'None' && <span className="text-sm text-slate-400 ml-1">TP {best.tp_pct}</span>}
                  </div>
                  <div className="flex items-center gap-1 mt-0.5">
                    <span className={`text-xs font-mono ${improved ? 'text-emerald-400' : 'text-red-400'}`}>
                      Sharpe {best?.sharpe_ratio?.toFixed(2) || '—'}
                    </span>
                    {improved && <Zap className="h-3 w-3 text-emerald-400" />}
                    <span className="text-[10px] text-slate-600 ml-1">
                      (baseline: {base?.sharpe_ratio?.toFixed(2) || '—'})
                    </span>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Metric selector */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm text-slate-500">Metric:</span>
          {METRICS.map(m => (
            <button
              key={m.key}
              onClick={() => setMetric(m.key)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                metric === m.key
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  : 'bg-[#111827] text-slate-400 border border-slate-800 hover:border-slate-600'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        <Tabs defaultValue="heatmap" className="space-y-4">
          <TabsList className="bg-[#111827] border border-slate-800 p-1">
            <TabsTrigger value="heatmap" className="data-[state=active]:bg-slate-700 data-[state=active]:text-amber-400">
              Heatmap
            </TabsTrigger>
            <TabsTrigger value="lines" className="data-[state=active]:bg-slate-700 data-[state=active]:text-amber-400">
              Line Charts
            </TabsTrigger>
            <TabsTrigger value="table" className="data-[state=active]:bg-slate-700 data-[state=active]:text-amber-400">
              Full Table
            </TabsTrigger>
          </TabsList>

          {/* ─── Heatmap ─── */}
          <TabsContent value="heatmap">
            <Card className="bg-[#111827] border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">{metricConfig.label} × Stop-Loss Level</CardTitle>
                <CardDescription className="text-slate-500">
                  Each cell shows {metricConfig.label.toLowerCase()} for the given algorithm and SL setting
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-3 px-2 text-slate-400 font-medium">Algorithm</th>
                        <th className="text-center py-3 px-1 text-slate-500 text-xs font-medium">No SL</th>
                        {data.sl_levels.map(sl => (
                          <th key={sl} className="text-center py-3 px-1 text-slate-500 text-xs font-medium">{sl}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {algoNames.map(algo => {
                        const baseVal = (data.no_sl_baseline[algo] as any)?.[metric] ?? 0
                        return (
                          <tr key={algo} className="border-b border-slate-800/50">
                            <td className="py-2 px-2 font-medium flex items-center gap-2">
                              <span className="w-2 h-2 rounded-full" style={{ background: ALGO_COLORS[algo] }} />
                              <span className="text-slate-200">{algo}</span>
                            </td>
                            {/* Baseline cell */}
                            <td className="py-2 px-1 text-center">
                              <div className="inline-block px-2 py-1 rounded text-xs font-mono bg-slate-700/50 text-slate-300">
                                {metricConfig.fmt(baseVal)}
                              </div>
                            </td>
                            {/* SL cells */}
                            {data.sl_levels.map(sl => {
                              const cell = data.grid[algo]?.[sl]?.[tpLabel]
                              const val = cell ? (cell as any)[metric] : 0
                              const bg = metric === 'max_drawdown'
                                ? sharpeColor(heatMax - val + heatMin, heatMin, heatMax)
                                : sharpeColor(val, heatMin, heatMax)
                              return (
                                <td key={sl} className="py-2 px-1 text-center">
                                  <div
                                    className="inline-block px-2 py-1 rounded text-xs font-mono font-medium min-w-[56px]"
                                    style={{ background: bg, color: textColor(bg) }}
                                    title={`${algo} | SL ${sl} | ${metricConfig.label}: ${metricConfig.fmt(val)}`}
                                  >
                                    {metricConfig.fmt(val)}
                                  </div>
                                </td>
                              )
                            })}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                {/* Legend */}
                <div className="flex items-center gap-2 mt-4 justify-center">
                  <span className="text-[11px] text-slate-500">Worse</span>
                  <div className="flex h-3 rounded overflow-hidden">
                    {Array.from({ length: 20 }, (_, i) => (
                      <div key={i} className="w-3 h-3" style={{ background: sharpeColor(i / 19, 0, 1) }} />
                    ))}
                  </div>
                  <span className="text-[11px] text-slate-500">Better</span>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ─── Line Charts ─── */}
          <TabsContent value="lines" className="space-y-4">
            {METRICS.map(m => (
              <Card key={m.key} className="bg-[#111827] border-slate-800">
                <CardHeader className="pb-2">
                  <CardTitle className="text-slate-100 text-lg">{m.label} vs Stop-Loss %</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart
                      data={data.sl_levels.map(sl => {
                        const point: Record<string, any> = { sl }
                        for (const algo of algoNames) {
                          const cell = data.grid[algo]?.[sl]?.[tpLabel]
                          if (cell) point[algo] = (cell as any)[m.key]
                        }
                        return point
                      })}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="sl" stroke="#64748b" tick={{ fontSize: 12 }} label={{ value: 'Stop-Loss %', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 12 }} />
                      <YAxis stroke="#334155" tickFormatter={(v) => typeof v === 'number' ? v.toFixed(1) : v} />
                      <Tooltip
                        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0' }}
                        formatter={(v: number, name: string) => [m.fmt(v), name]}
                      />
                      <Legend />
                      {algoNames.map(algo => {
                        const baseVal = (data.no_sl_baseline[algo] as any)?.[m.key]
                        return (
                          <Line
                            key={algo}
                            type="monotone"
                            dataKey={algo}
                            stroke={ALGO_COLORS[algo] || '#888'}
                            strokeWidth={2}
                            dot={{ r: 3 }}
                            activeDot={{ r: 5 }}
                          />
                        )
                      })}
                      {/* Baseline reference lines */}
                      {algoNames.map(algo => {
                        const baseVal = (data.no_sl_baseline[algo] as any)?.[m.key]
                        if (baseVal == null) return null
                        return (
                          <ReferenceLine
                            key={`ref-${algo}`}
                            y={baseVal}
                            stroke={ALGO_COLORS[algo]}
                            strokeDasharray="4 4"
                            strokeOpacity={0.4}
                          />
                        )
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-[11px] text-slate-600 text-center mt-1">
                    Dashed lines = baseline (no stop-loss)
                  </p>
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          {/* ─── Full Table ─── */}
          <TabsContent value="table">
            <Card className="bg-[#111827] border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">Complete Results Grid</CardTitle>
                <CardDescription className="text-slate-500">
                  All metrics for every algorithm × SL combination
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-700 text-slate-400">
                        <th className="text-left py-3 px-2">Algorithm</th>
                        <th className="text-center py-3 px-2">SL %</th>
                        <th className="text-right py-3 px-2">Return</th>
                        <th className="text-right py-3 px-2">Sharpe</th>
                        <th className="text-right py-3 px-2">Max DD</th>
                        <th className="text-right py-3 px-2">Win Rate</th>
                        <th className="text-right py-3 px-2">Trades</th>
                        <th className="text-right py-3 px-2">SL Hits</th>
                        <th className="text-right py-3 px-2">TP Hits</th>
                        <th className="text-right py-3 px-2">Final $</th>
                      </tr>
                    </thead>
                    <tbody>
                      {algoNames.flatMap(algo => {
                        const base = data.no_sl_baseline[algo]
                        const bestSharpe = data.best_per_algo[algo]?.sharpe_ratio
                        return [
                          // Baseline row
                          <tr key={`${algo}-base`} className="border-b border-slate-800/30 bg-slate-800/20">
                            <td className="py-2 px-2 font-medium flex items-center gap-1.5">
                              <span className="w-1.5 h-1.5 rounded-full" style={{ background: ALGO_COLORS[algo] }} />
                              <span className="text-slate-200">{algo}</span>
                            </td>
                            <td className="text-center py-2 px-2 text-slate-500 italic">None</td>
                            <td className={`text-right py-2 px-2 font-mono ${base.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {base.total_return >= 0 ? '+' : ''}{base.total_return.toFixed(2)}%
                            </td>
                            <td className="text-right py-2 px-2 font-mono text-slate-300">{base.sharpe_ratio.toFixed(2)}</td>
                            <td className="text-right py-2 px-2 font-mono text-red-400">-{base.max_drawdown.toFixed(2)}%</td>
                            <td className="text-right py-2 px-2 font-mono text-slate-300">{base.win_rate.toFixed(1)}%</td>
                            <td className="text-right py-2 px-2 font-mono text-slate-400">{base.total_trades}</td>
                            <td className="text-right py-2 px-2 font-mono text-slate-500">—</td>
                            <td className="text-right py-2 px-2 font-mono text-slate-500">—</td>
                            <td className="text-right py-2 px-2 font-mono text-slate-400">${base.final_balance.toLocaleString()}</td>
                          </tr>,
                          // SL rows
                          ...data.sl_levels.map(sl => {
                            const m = data.grid[algo]?.[sl]?.[tpLabel]
                            if (!m) return null
                            const isBest = m.sharpe_ratio === bestSharpe && m.sl_pct === data.best_per_algo[algo]?.sl_pct
                            return (
                              <tr key={`${algo}-${sl}`} className={`border-b border-slate-800/20 ${isBest ? 'bg-emerald-500/5' : ''}`}>
                                <td className="py-2 px-2"></td>
                                <td className="text-center py-2 px-2 text-slate-300 font-mono">
                                  {sl}
                                  {isBest && <Badge className="ml-1 text-[9px] bg-emerald-500/20 text-emerald-400 border-0 py-0">BEST</Badge>}
                                </td>
                                <td className={`text-right py-2 px-2 font-mono ${m.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                  {m.total_return >= 0 ? '+' : ''}{m.total_return.toFixed(2)}%
                                </td>
                                <td className="text-right py-2 px-2 font-mono text-slate-300">{m.sharpe_ratio.toFixed(2)}</td>
                                <td className="text-right py-2 px-2 font-mono text-red-400">-{m.max_drawdown.toFixed(2)}%</td>
                                <td className="text-right py-2 px-2 font-mono text-slate-300">{m.win_rate.toFixed(1)}%</td>
                                <td className="text-right py-2 px-2 font-mono text-slate-400">{m.total_trades}</td>
                                <td className="text-right py-2 px-2 font-mono text-amber-400">{m.sl_hits}</td>
                                <td className="text-right py-2 px-2 font-mono text-emerald-400">{m.tp_hits}</td>
                                <td className="text-right py-2 px-2 font-mono text-slate-400">${m.final_balance.toLocaleString()}</td>
                              </tr>
                            )
                          }).filter(Boolean),
                        ]
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
