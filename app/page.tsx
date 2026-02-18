"use client"

import { useState, useEffect, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar, Legend,
  ReferenceLine, Area, AreaChart
} from "recharts"
import { TrendingUp, TrendingDown, DollarSign, Activity, Brain, Target, BarChart3, Zap, RefreshCw } from "lucide-react"

// ── types ────────────────────────────────────────────────────────

type AlgoResult = {
  algorithm: string
  total_return: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  total_trades: number
  avg_trade_pnl: number
  final_balance: number
  equity_curve: number[]
  trades: {
    step: number
    timestamp: string
    action: string
    price: number
    size: number
    pnl?: number
    fee: number
  }[]
}

type Results = Record<string, AlgoResult>

const ALGO_COLORS: Record<string, string> = {
  "DQN":         "#f59e0b",
  "Double DQN":  "#3b82f6",
  "Dueling DQN": "#10b981",
  "A2C":         "#8b5cf6",
  "Buy & Hold":  "#6b7280",
}

// ── helpers ──────────────────────────────────────────────────────

function fmt(n: number, d = 2) { return n.toFixed(d) }
function fmtUsd(n: number) { return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}` }

function bestAlgo(results: Results): AlgoResult | null {
  const algos = Object.values(results).filter(r => r.algorithm !== "Buy & Hold")
  if (!algos.length) return null
  return algos.reduce((best, cur) =>
    cur.sharpe_ratio > best.sharpe_ratio ? cur : best
  )
}

// ── component ───────────────────────────────────────────────────

export default function Dashboard() {
  const [results, setResults] = useState<Results | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchResults = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/results")
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.error || "Failed to load results")
      }
      setResults(await res.json())
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchResults() }, [fetchResults])

  // ── loading / error states ──────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0e17] flex items-center justify-center">
        <div className="text-center space-y-4">
          <RefreshCw className="h-10 w-10 text-emerald-400 animate-spin mx-auto" />
          <p className="text-slate-400 text-lg">Loading backtest results…</p>
        </div>
      </div>
    )
  }

  if (error || !results) {
    return (
      <div className="min-h-screen bg-[#0a0e17] flex items-center justify-center p-6">
        <Card className="max-w-lg w-full bg-[#111827] border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <Brain className="h-6 w-6 text-emerald-400" />
              No Backtest Results
            </CardTitle>
            <CardDescription className="text-slate-400">
              Run the training pipeline first to generate results.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="bg-[#0a0e17] rounded-lg p-4 font-mono text-sm text-slate-300 space-y-1">
              <p className="text-slate-500"># 1. Download data (or generate sample)</p>
              <p>python scripts/fetch_data.py</p>
              <p className="text-slate-500 mt-2"># — or generate synthetic test data —</p>
              <p>python scripts/generate_sample_data.py</p>
              <p className="text-slate-500 mt-2"># 2. Train all 4 models + backtest</p>
              <p>python scripts/train_all.py --episodes 50</p>
              <p className="text-slate-500 mt-2"># 3. Refresh this page</p>
            </div>
            <Button onClick={fetchResults} variant="outline" className="w-full border-slate-600 text-slate-300 hover:bg-slate-700">
              <RefreshCw className="h-4 w-4 mr-2" /> Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  // ── data prep ───────────────────────────────────────────────
  const best = bestAlgo(results)!
  const algoNames = Object.keys(results).filter(k => k !== "Buy & Hold")
  const bh = results["Buy & Hold"]

  // Equity curve chart data — align all curves by index
  const maxLen = Math.max(...Object.values(results).map(r => r.equity_curve.length))
  const equityData = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, any> = { idx: i }
    for (const [name, r] of Object.entries(results)) {
      const idx = Math.min(i, r.equity_curve.length - 1)
      point[name] = r.equity_curve[idx]
    }
    return point
  })

  // Algorithm comparison bar data
  const comparisonData = Object.values(results)
    .filter(r => r.algorithm !== "Buy & Hold")
    .map(r => ({
      algorithm: r.algorithm,
      "Total Return (%)": r.total_return,
      "Sharpe Ratio": r.sharpe_ratio,
      "Win Rate (%)": r.win_rate,
    }))

  // Drawdown comparison
  const ddData = Object.values(results).map(r => ({
    algorithm: r.algorithm,
    drawdown: -r.max_drawdown,
  }))

  // Recent trades from all algos
  const allTrades = Object.values(results)
    .flatMap(r => r.trades.map(t => ({ ...t, algo: r.algorithm })))
    .filter(t => t.timestamp)
    .sort((a, b) => b.step - a.step)
    .slice(0, 20)

  // ── render ──────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-[#0a0e17] text-slate-100 p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/10">
              <Brain className="h-8 w-8 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
                DRL Crypto Backtester
              </h1>
              <p className="text-slate-500 text-sm">
                SOL/USDT 15m · 4 algorithms · Comparative study
              </p>
            </div>
          </div>
          <Button
            onClick={fetchResults}
            variant="outline"
            size="sm"
            className="border-slate-700 text-slate-400 hover:text-slate-100 hover:bg-slate-800"
          >
            <RefreshCw className="h-4 w-4 mr-1" /> Refresh
          </Button>
        </div>

        {/* KPI Row */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <MetricCard
            label="Best Algorithm"
            value={best.algorithm}
            sub={`Sharpe ${fmt(best.sharpe_ratio)}`}
            icon={<Target className="h-4 w-4" />}
            accent="emerald"
          />
          <MetricCard
            label="Best Return"
            value={`${fmt(best.total_return)}%`}
            sub={fmtUsd(best.final_balance)}
            icon={<TrendingUp className="h-4 w-4" />}
            accent={best.total_return >= 0 ? "emerald" : "red"}
          />
          <MetricCard
            label="Best Sharpe"
            value={fmt(best.sharpe_ratio)}
            sub="Risk-adjusted"
            icon={<Activity className="h-4 w-4" />}
            accent="blue"
          />
          <MetricCard
            label="Best Win Rate"
            value={`${fmt(best.win_rate, 1)}%`}
            sub={`${best.total_trades} trades`}
            icon={<Zap className="h-4 w-4" />}
            accent="amber"
          />
          <MetricCard
            label="Buy & Hold"
            value={bh ? `${fmt(bh.total_return)}%` : "—"}
            sub="Baseline"
            icon={<DollarSign className="h-4 w-4" />}
            accent="slate"
          />
        </div>

        {/* Tabs */}
        <Tabs defaultValue="equity" className="space-y-4">
          <TabsList className="bg-[#111827] border border-slate-800 p-1">
            <TabsTrigger value="equity" className="data-[state=active]:bg-slate-700 data-[state=active]:text-emerald-400">
              Equity Curves
            </TabsTrigger>
            <TabsTrigger value="compare" className="data-[state=active]:bg-slate-700 data-[state=active]:text-emerald-400">
              Comparison
            </TabsTrigger>
            <TabsTrigger value="trades" className="data-[state=active]:bg-slate-700 data-[state=active]:text-emerald-400">
              Trades
            </TabsTrigger>
            <TabsTrigger value="details" className="data-[state=active]:bg-slate-700 data-[state=active]:text-emerald-400">
              Details
            </TabsTrigger>
          </TabsList>

          {/* ─── Equity Curves ─── */}
          <TabsContent value="equity">
            <Card className="bg-[#111827] border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">Portfolio Equity Curves</CardTitle>
                <CardDescription className="text-slate-500">
                  All strategies vs Buy &amp; Hold on the test period
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={420}>
                  <LineChart data={equityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="idx" tick={false} stroke="#334155" />
                    <YAxis
                      stroke="#334155"
                      tickFormatter={(v: number) => `$${(v/1000).toFixed(1)}k`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: '#1e293b', border: '1px solid #334155',
                        borderRadius: 8, color: '#e2e8f0',
                      }}
                      formatter={(v: number) => [`$${v.toLocaleString()}`, undefined]}
                    />
                    <Legend />
                    {Object.keys(results).map(name => (
                      <Line
                        key={name}
                        type="monotone"
                        dataKey={name}
                        stroke={ALGO_COLORS[name] || "#888"}
                        strokeWidth={name === best.algorithm ? 2.5 : 1.5}
                        dot={false}
                        opacity={name === "Buy & Hold" ? 0.5 : 1}
                        strokeDasharray={name === "Buy & Hold" ? "6 3" : undefined}
                      />
                    ))}
                    <ReferenceLine y={10000} stroke="#475569" strokeDasharray="3 3" label="" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ─── Comparison ─── */}
          <TabsContent value="compare" className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              {/* Returns bar chart */}
              <Card className="bg-[#111827] border-slate-800">
                <CardHeader>
                  <CardTitle className="text-slate-100 text-lg">Returns &amp; Win Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={comparisonData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="algorithm" stroke="#64748b" tick={{ fontSize: 12 }} />
                      <YAxis stroke="#334155" />
                      <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0' }} />
                      <Legend />
                      <Bar dataKey="Total Return (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Win Rate (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Drawdown chart */}
              <Card className="bg-[#111827] border-slate-800">
                <CardHeader>
                  <CardTitle className="text-slate-100 text-lg">Max Drawdown</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={ddData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis type="number" stroke="#334155" tickFormatter={(v) => `${v}%`} />
                      <YAxis type="category" dataKey="algorithm" stroke="#64748b" width={100} tick={{ fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0' }}
                        formatter={(v: number) => [`${v.toFixed(2)}%`, "Drawdown"]}
                      />
                      <Bar dataKey="drawdown" fill="#ef4444" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Summary table */}
            <Card className="bg-[#111827] border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100 text-lg">Full Metrics Table</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700 text-slate-400">
                        <th className="text-left py-3 px-2">Algorithm</th>
                        <th className="text-right py-3 px-2">Return</th>
                        <th className="text-right py-3 px-2">Sharpe</th>
                        <th className="text-right py-3 px-2">Max DD</th>
                        <th className="text-right py-3 px-2">Win Rate</th>
                        <th className="text-right py-3 px-2">Trades</th>
                        <th className="text-right py-3 px-2">Final Balance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.values(results).map(r => (
                        <tr
                          key={r.algorithm}
                          className={`border-b border-slate-800 ${r.algorithm === best.algorithm ? 'bg-emerald-500/5' : ''}`}
                        >
                          <td className="py-3 px-2 font-medium flex items-center gap-2">
                            <span
                              className="w-2 h-2 rounded-full inline-block"
                              style={{ background: ALGO_COLORS[r.algorithm] }}
                            />
                            {r.algorithm}
                            {r.algorithm === best.algorithm && (
                              <Badge className="text-[10px] bg-emerald-500/20 text-emerald-400 border-0">
                                BEST
                              </Badge>
                            )}
                          </td>
                          <td className={`text-right py-3 px-2 font-mono ${r.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {r.total_return >= 0 ? '+' : ''}{fmt(r.total_return)}%
                          </td>
                          <td className="text-right py-3 px-2 font-mono text-slate-300">{fmt(r.sharpe_ratio)}</td>
                          <td className="text-right py-3 px-2 font-mono text-red-400">-{fmt(r.max_drawdown)}%</td>
                          <td className="text-right py-3 px-2 font-mono text-slate-300">{fmt(r.win_rate, 1)}%</td>
                          <td className="text-right py-3 px-2 font-mono text-slate-400">{r.total_trades}</td>
                          <td className="text-right py-3 px-2 font-mono text-slate-300">{fmtUsd(r.final_balance)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ─── Trades ─── */}
          <TabsContent value="trades">
            <Card className="bg-[#111827] border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">Recent Trades</CardTitle>
                <CardDescription className="text-slate-500">
                  Last 20 trades across all strategies
                </CardDescription>
              </CardHeader>
              <CardContent>
                {allTrades.length === 0 ? (
                  <p className="text-slate-500 py-6 text-center">No trade data available.</p>
                ) : (
                  <div className="space-y-2">
                    {allTrades.map((t, i) => (
                      <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-[#0a0e17] border border-slate-800">
                        <div className="flex items-center gap-3">
                          <Badge
                            className={
                              t.action === "LONG"  ? "bg-emerald-500/20 text-emerald-400 border-0" :
                              t.action === "SHORT" ? "bg-red-500/20 text-red-400 border-0" :
                              "bg-slate-500/20 text-slate-400 border-0"
                            }
                          >
                            {t.action}
                          </Badge>
                          <div>
                            <span className="text-sm font-medium text-slate-200">{t.algo}</span>
                            <span className="text-xs text-slate-500 ml-2">
                              {t.timestamp ? new Date(t.timestamp).toLocaleString() : `step ${t.step}`}
                            </span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-mono text-slate-200">
                            ${t.price.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                          </div>
                          {t.pnl !== undefined && (
                            <div className={`text-xs font-mono ${t.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {t.pnl >= 0 ? '+' : ''}{t.pnl.toFixed(2)} PnL
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* ─── Details ─── */}
          <TabsContent value="details" className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              {algoNames.map(name => {
                const r = results[name]
                return (
                  <Card key={name} className="bg-[#111827] border-slate-800">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full" style={{ background: ALGO_COLORS[name] }} />
                        <span className="text-slate-100">{name}</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <MiniStat label="Return" value={`${r.total_return >= 0 ? '+' : ''}${fmt(r.total_return)}%`}
                          color={r.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'} />
                        <MiniStat label="Sharpe" value={fmt(r.sharpe_ratio)} color="text-blue-400" />
                        <MiniStat label="Max DD" value={`-${fmt(r.max_drawdown)}%`} color="text-red-400" />
                        <MiniStat label="Win Rate" value={`${fmt(r.win_rate, 1)}%`} color="text-amber-400" />
                        <MiniStat label="Trades" value={String(r.total_trades)} color="text-slate-300" />
                        <MiniStat label="Avg PnL" value={`$${r.avg_trade_pnl.toFixed(2)}`}
                          color={r.avg_trade_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'} />
                      </div>
                      <div className="h-24">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={r.equity_curve.filter((_, i) => i % Math.max(1, Math.floor(r.equity_curve.length / 200)) === 0).map((v, i) => ({ i, v }))}>
                            <Area
                              type="monotone" dataKey="v"
                              stroke={ALGO_COLORS[name]}
                              fill={ALGO_COLORS[name]}
                              fillOpacity={0.1}
                              strokeWidth={1.5}
                              dot={false}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>

            {/* Architecture info */}
            <Card className="bg-[#111827] border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100 text-lg">Algorithm Architecture</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <ArchInfo
                    title="DQN"
                    desc="Vanilla Deep Q-Network. Uses a target network updated periodically. Both action selection and evaluation use the target network."
                    color="#f59e0b"
                  />
                  <ArchInfo
                    title="Double DQN"
                    desc="Decouples action selection (online network) from evaluation (target network), reducing overestimation bias in Q-values."
                    color="#3b82f6"
                  />
                  <ArchInfo
                    title="Dueling DQN"
                    desc="Separate Value V(s) and Advantage A(s,a) streams. Q(s,a) = V(s) + A(s,a) - mean(A). Better state-value estimation."
                    color="#10b981"
                  />
                  <ArchInfo
                    title="A2C"
                    desc="Advantage Actor-Critic. On-policy method with shared feature trunk. Actor outputs action probabilities, critic estimates state value."
                    color="#8b5cf6"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

// ── sub-components ───────────────────────────────────────────────

function MetricCard({ label, value, sub, icon, accent }: {
  label: string; value: string; sub: string; icon: React.ReactNode
  accent: "emerald" | "blue" | "amber" | "red" | "slate"
}) {
  const colors = {
    emerald: "text-emerald-400", blue: "text-blue-400",
    amber: "text-amber-400", red: "text-red-400", slate: "text-slate-400",
  }
  return (
    <Card className="bg-[#111827] border-slate-800">
      <CardContent className="pt-4 pb-3 px-4">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
          <span className="text-slate-600">{icon}</span>
        </div>
        <div className={`text-xl font-bold ${colors[accent]}`}>{value}</div>
        <div className="text-xs text-slate-500 mt-0.5">{sub}</div>
      </CardContent>
    </Card>
  )
}

function MiniStat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-[#0a0e17] rounded-lg p-2">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className={`text-sm font-mono font-medium ${color}`}>{value}</div>
    </div>
  )
}

function ArchInfo({ title, desc, color }: { title: string; desc: string; color: string }) {
  return (
    <div className="p-3 rounded-lg bg-[#0a0e17] border border-slate-800">
      <div className="flex items-center gap-2 mb-1">
        <span className="w-2 h-2 rounded-full" style={{ background: color }} />
        <span className="font-medium text-slate-200">{title}</span>
      </div>
      <p className="text-slate-500 text-xs leading-relaxed">{desc}</p>
    </div>
  )
}
