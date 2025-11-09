import Layout from "@/components/Layout";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { TrendingUp, TrendingDown, Package, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { fetchAnalyticsSeries, type AnalyticsSeriesPoint, type AnalyticsSeriesResponse } from "@/lib/backend-api";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const Insights = () => {
  const [insights, setInsights] = useState<AnalyticsSeriesResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [timeRange, setTimeRange] = useState<"weekly" | "monthly">("weekly");
  const [selectedItem, setSelectedItem] = useState<string>("general");
  const [availableItems, setAvailableItems] = useState<string[]>(["general"]);
  const { toast } = useToast();

  const fetchAnalytics = async (range: "weekly" | "monthly", item: string) => {
    setIsLoading(true);
    try {
      const response = await fetchAnalyticsSeries(range, item);
      setInsights(response);

      if (response.availableItems && response.availableItems.length > 0) {
        setAvailableItems(response.availableItems);
        if (!response.availableItems.includes(item)) {
          setSelectedItem(response.availableItems[0]);
        }
      } else {
        setAvailableItems(["general"]);
      }
    } catch (error) {
      console.error("Error fetching analytics:", error);
      toast({
        title: "Failed to load insights",
        description: "Please try again later",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics(timeRange, selectedItem);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeRange, selectedItem]);

  const chartData: AnalyticsSeriesPoint[] = insights?.series ?? [];
  const totals = insights?.totals ?? {};

  const chartConfig = [
    {
      dataKey: "spent" as const,
      label: "Spent",
      color: "#0f172a",
    },
    {
      dataKey: "saved" as const,
      label: "Saved",
      color: "#16a34a",
    },
    {
      dataKey: "wasted" as const,
      label: "Wasted",
      color: "#f97316",
    },
  ];

  const formatItemLabel = (value: string) => {
    if (!value) {
      return "";
    }
    if (value.toLowerCase() === "general") {
      return "General";
    }
    return value
      .split(/\s+/)
      .filter(Boolean)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const formatXAxisTick = (value: string) => {
    if (timeRange === "monthly") {
      const [year, month] = value.split("-");
      if (!month || !year) {
        return value;
      }
      const monthIndex = Number.parseInt(month, 10);
      const date = new Date(Number.parseInt(year, 10), monthIndex - 1, 1);
      return date.toLocaleString(undefined, { month: "short" });
    }
    if (value.length >= 10) {
      return value.slice(5);
    }
    return value;
  };

  const formatTooltipLabel = (label: string, payload?: any[]) => {
    const point = payload && payload[0]?.payload;
    if (!point) {
      return label;
    }
    if (timeRange === "monthly") {
      const [year, month] = label.split("-");
      const date = new Date(Number.parseInt(year ?? "0", 10), Number.parseInt(month ?? "1", 10) - 1, 1);
      return date.toLocaleString(undefined, { month: "long", year: "numeric" });
    }
    const start = point.rangeStart ?? label;
    const end = point.rangeEnd ?? label;
    return `Week ${start} → ${end}`;
  };

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-foreground mb-2">Insights</h1>
          <p className="text-muted-foreground">Analyze your spending patterns</p>
        </header>

        <div className="flex flex-col gap-4 mb-6 md:flex-row md:items-center">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Time Range</span>
            <Select value={timeRange} onValueChange={(value) => setTimeRange(value as "weekly" | "monthly")}>
              <SelectTrigger className="w-36">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="weekly">Weekly</SelectItem>
                <SelectItem value="monthly">Monthly</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Item</span>
            <Select value={selectedItem} onValueChange={(value) => setSelectedItem(value)}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {availableItems.map((item) => (
                  <SelectItem key={item} value={item}>
                    {formatItemLabel(item)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : insights && chartData.length > 0 ? (
          <>
            {/* Summary Stats */}
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <Card className="p-4 bg-warning-light border-warning/20">
                <div className="flex items-center gap-3 mb-2">
                  <TrendingDown className="w-5 h-5 text-warning" />
                  <span className="text-sm font-medium text-muted-foreground">Waste Rate</span>
                </div>
                <p className="text-2xl font-bold text-warning">{(totals.wastePercentage ?? 0).toFixed(1)}%</p>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-3 mb-2">
                  <Package className="w-5 h-5 text-primary" />
                  <span className="text-sm font-medium text-muted-foreground">Total Spent</span>
                </div>
                <p className="text-2xl font-bold text-foreground">€{(totals.spent ?? 0).toFixed(2)}</p>
              </Card>

              <Card className="p-4 bg-success-light border-success/20">
                <div className="flex items-center gap-3 mb-2">
                  <TrendingUp className="w-5 h-5 text-success" />
                  <span className="text-sm font-medium text-muted-foreground">Total Saved</span>
                </div>
                <p className="text-2xl font-bold text-success">€{(totals.saved ?? 0).toFixed(2)}</p>
              </Card>
            </div>

            <Card className="p-6">
              <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-2 mb-4">
                <div>
                  <h2 className="text-xl font-semibold text-foreground">Spending vs Savings vs Waste</h2>
                  <p className="text-sm text-muted-foreground">
                    Daily totals for the selected time range and item
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: "#0f172a" }} />
                    <span>Spent</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: "#16a34a" }} />
                    <span>Saved</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: "#f97316" }} />
                    <span>Wasted</span>
                  </div>
                </div>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" />
                    <XAxis dataKey="date" tick={{ fontSize: 12 }} tickFormatter={formatXAxisTick} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip
                      formatter={(value: number) => `€${value.toFixed(2)}`}
                      labelFormatter={(label, payload) => formatTooltipLabel(String(label), payload)}
                    />
                    <Legend />
                    {chartConfig.map((chart) => (
                      <Line
                        key={chart.dataKey}
                        type="monotone"
                        dataKey={chart.dataKey}
                        name={chart.label}
                        stroke={chart.color}
                        strokeWidth={2}
                        dot={{ r: 4 }}
                        activeDot={{ r: 6 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <div className="mt-6 p-4 rounded-lg bg-muted">
              <p className="text-sm text-muted-foreground text-center">
                📊 Analytics are calculated on-the-fly from your shopping data
              </p>
            </div>
          </>
        ) : !isLoading ? (
          <div className="text-center py-12 text-muted-foreground">
            No data available
          </div>
        ) : null}
      </div>
    </Layout>
  );
};

export default Insights;
