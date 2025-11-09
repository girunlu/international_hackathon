import { useEffect, useState } from "react";
import Layout from "@/components/Layout";
import { Card } from "@/components/ui/card";
import { ShoppingBag, Calendar } from "lucide-react";
import { getShoppingHistory } from "@/lib/backend-api";

interface ShoppingHistoryItem {
  id: number;
  date: string;
  items: Array<{
    name: string;
    amount: number;
    totalPrice: number;
    wastedPrice: number;
    weeklyTotalPrice?: number;
    weeklyWastedPrice?: number;
    itemWeeklyTotalPrice?: number;
    itemWeeklyWastedPrice?: number;
  }>;
  total: number;
}

const History = () => {
  const [shoppingTrips, setShoppingTrips] = useState<ShoppingHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        setIsLoading(true);
        const history = await getShoppingHistory();
        setShoppingTrips(history);
      } catch (error) {
        console.error("Error fetching shopping history:", error);
        setShoppingTrips([]);
      } finally {
        setIsLoading(false);
      }
    };

    loadHistory();
  }, []);

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-foreground mb-2">Shopping History</h1>
          <p className="text-muted-foreground">Track your past purchases and waste</p>
        </header>

        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">Loading shopping history...</div>
        ) : shoppingTrips.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <ShoppingBag className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No shopping history found.</p>
            <p className="text-sm">Start logging your purchases to see them here.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {shoppingTrips.map((trip) => (
              <Card key={trip.id} className="p-5">
                <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border">
                  <Calendar className="w-5 h-5 text-primary" />
                  <span className="font-semibold text-foreground">{trip.date}</span>
                  <span className="ml-auto text-lg font-bold text-foreground">€{trip.total.toFixed(2)}</span>
                </div>

                <div className="space-y-2">
                  {trip.items.map((item, idx) => {
                    const weeklyWaste =
                      item.itemWeeklyWastedPrice ??
                      item.wastedPrice ??
                      item.weeklyWastedPrice ??
                      0;
                    const weeklyTotal =
                      item.itemWeeklyTotalPrice ??
                      item.totalPrice ??
                      item.weeklyTotalPrice ??
                      0;
                    const hasWaste = weeklyWaste > 0;
                    const wastePercentage =
                      hasWaste && weeklyTotal > 0
                        ? ((weeklyWaste / weeklyTotal) * 100).toFixed(0)
                        : "0";

                    return (
                      <div
                        key={idx}
                        className={`flex items-center justify-between p-3 rounded-lg ${
                          hasWaste ? "bg-destructive/10 border border-destructive/20" : "bg-accent/50"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <ShoppingBag className={`w-4 h-4 ${hasWaste ? "text-destructive" : "text-primary"}`} />
                          <div className="flex flex-col">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-foreground">{item.name}</span>
                              <span className="text-sm text-muted-foreground">×{item.amount}</span>
                            </div>
                            {hasWaste && (
                              <span className="text-xs text-destructive font-medium">
                                €{weeklyWaste.toFixed(2)} wasted / €{weeklyTotal.toFixed(2)} ({wastePercentage}%)
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className={`text-sm font-semibold ${hasWaste ? "text-destructive" : "text-foreground"}`}>
                            €{item.totalPrice.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </Layout>
  );
};

export default History;
