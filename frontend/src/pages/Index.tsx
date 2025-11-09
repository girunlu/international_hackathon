import Layout from "@/components/Layout";
import { Card } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Loader2 } from "lucide-react";
import heroWallet from "@/assets/hero-wallet.png";
import { useState, useEffect } from "react";
import { getAnalyticsSummary, getWeeklyRecommendations } from "@/lib/backend-api";

const Index = () => {
  const [stats, setStats] = useState({
    totalSaved: 0,
    totalWasted: 0,
    totalSpent: 0,
    wastePercentage: 0,
  });
  const [mostBought, setMostBought] = useState<Array<{ name: string; count: number; amount: number }>>([]);
  const [topSavings, setTopSavings] = useState<Array<{ name: string; amount: number; count: number }>>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [recommendation, setRecommendation] = useState<string>("");

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setIsLoading(true);
        const [analytics, weeklyRecommendation] = await Promise.all([
          getAnalyticsSummary(),
          getWeeklyRecommendations(),
        ]);

        if (analytics) {
          setStats({
            totalSaved: analytics.totalSaved ?? 0,
            totalWasted: analytics.totalWasted ?? 0,
            totalSpent: analytics.totalSpent ?? 0,
            wastePercentage: analytics.wastePercentage ?? 0,
          });

          setMostBought(
            (analytics.topWasted ?? []).map((item) => ({
              name: item.name,
              count: item.count,
              amount: item.amount,
            })),
          );

          setTopSavings(
            (analytics.topSaved ?? []).map((item) => ({
              name: item.name,
              amount: item.amount,
              count: item.count,
            })),
          );
        }

        if (weeklyRecommendation?.recommendation) {
          setRecommendation(weeklyRecommendation.recommendation);
        } else {
          setRecommendation("");
        }
      } catch (error) {
        console.error("Error fetching dashboard data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  return (
    <Layout>
      {/* Hero Section - Full Width */}
      <div className="relative min-h-[30vh] flex items-center justify-center mb-8 overflow-hidden">
        <div
          className="absolute inset-0 bg-gradient-to-br from-primary/10 via-background to-accent/20"
          style={{
            backgroundImage: `url(${heroWallet})`,
            backgroundSize: "cover",
            backgroundPosition: "center 75%",
            opacity: 0.65,
          }}
        />
        <div className="absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-background to-transparent" />
        <div className="relative container mx-auto px-4 max-w-4xl text-center z-10">
          <h1 className="text-5xl md:text-6xl font-bold text-foreground mb-4">Smart Shopping Tracker</h1>
          <p className="text-xl text-muted-foreground">Track your expenses, reduce waste, save money</p>
        </div>
      </div>

      <div className="container mx-auto px-4 max-w-4xl">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="grid md:grid-cols-3 gap-4 mb-8">
              <Card className="p-6 bg-success-light border-success/20">
                <div className="flex items-center gap-3 mb-2">
                  <TrendingUp className="w-6 h-6 text-success" />
                  <span className="text-sm font-medium text-muted-foreground">Total Saved (This Month)</span>
                </div>
                <p className="text-3xl font-bold text-success">€{stats.totalSaved.toFixed(2)}</p>
              </Card>

              <Card className="p-6 bg-warning-light border-warning/20">
                <div className="flex items-center gap-3 mb-2">
                  <TrendingDown className="w-6 h-6 text-warning" />
                  <span className="text-sm font-medium text-muted-foreground">Total Wasted (This Month)</span>
                </div>
                <p className="text-3xl font-bold text-warning">€{stats.totalWasted.toFixed(2)}</p>
              </Card>

              <Card className="p-6">
                <div className="flex items-center gap-3 mb-2">
                  <TrendingUp className="w-6 h-6 text-primary" />
                  <span className="text-sm font-medium text-muted-foreground">Total Spent (This Month)</span>
                </div>
                <p className="text-3xl font-bold text-primary">€{stats.totalSpent.toFixed(2)}</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Waste rate: {stats.wastePercentage.toFixed(1)}%
                </p>
              </Card>
            </div>

            {/* Lists Grid */}
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              {/* Most Bought Items */}
              <Card className="p-6">
                <h2 className="text-xl font-semibold text-foreground mb-4">Most Bought Items</h2>
                <div className="space-y-3">
                  {mostBought.map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-accent/50">
                      <div className="flex flex-col">
                        <span className="font-medium text-foreground">{item.name}</span>
                        <span className="text-xs text-muted-foreground">{item.count}× purchased</span>
                      </div>
                      <span className="text-sm font-semibold text-foreground">€{item.amount.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Top Savings */}
              <Card className="p-6">
                <h2 className="text-xl font-semibold text-foreground mb-4">Top Savings</h2>
                <div className="space-y-3">
                  {topSavings.map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-success-light">
                      <span className="font-medium text-foreground">{item.name}</span>
                      <div className="text-right">
                        <p className="text-sm font-bold text-success">€{item.amount.toFixed(2)}</p>
                        <p className="text-xs text-muted-foreground">{item.count}×</p>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            {/* Weekly Suggestion */}
            <Card className="p-6 bg-primary/5 border-primary/20">
              <h2 className="text-xl font-semibold text-foreground mb-3">💡 Weekly Suggestion</h2>
              <p className="text-muted-foreground leading-relaxed">
                {recommendation || "No recommendation available right now. Check back soon for fresh tips!"}
              </p>
            </Card>
          </>
        )}
      </div>
    </Layout>
  );
};

export default Index;
