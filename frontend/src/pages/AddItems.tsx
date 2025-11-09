import { useState, useEffect } from "react";
import Layout from "@/components/Layout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Trash2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { getFrequentItems, logWaste } from "@/lib/backend-api";
import { formatQuantityLabel } from "@/lib/utils";

const AddItems = () => {
  const [itemName, setItemName] = useState<string>("");
  const [quantityValue, setQuantityValue] = useState<string>("");
  const [quantityUnit, setQuantityUnit] = useState<string>("");
  const [frequentItems, setFrequentItems] = useState<string[]>([]);
  const [isLoadingFrequent, setIsLoadingFrequent] = useState<boolean>(true);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const { toast } = useToast();

  useEffect(() => {
    const loadFrequentItems = async () => {
      try {
        setIsLoadingFrequent(true);
        const items = await getFrequentItems();
        setFrequentItems(items);
      } catch (error) {
        console.error("Error fetching frequent items:", error);
        setFrequentItems([]);
      } finally {
        setIsLoadingFrequent(false);
      }
    };

    loadFrequentItems();
  }, []);

  const handleLogWaste = async () => {
    if (!itemName || !quantityValue) {
      toast({
        title: "Missing information",
        description: "Please fill in item name and quantity",
        variant: "destructive",
      });
      return;
    }

    const quantityNumeric = parseFloat(quantityValue);

    if (!Number.isFinite(quantityNumeric) || quantityNumeric <= 0) {
      toast({
        title: "Invalid quantity",
        description: "Quantity must be a positive number.",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsSubmitting(true);
      await logWaste(itemName, quantityNumeric, quantityUnit.trim() || null);

      toast({
        title: "Waste logged!",
        description: `${itemName} (${formatQuantityLabel(quantityNumeric, quantityUnit)}) logged as waste`,
      });

      // Reset form
      setItemName("");
      setQuantityValue("");
      setQuantityUnit("");
    } catch (error) {
      console.error("Error logging waste:", error);
      toast({
        title: "Failed to log waste",
        description: "Please try again later.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8 max-w-md">
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-foreground mb-2">Log Waste</h1>
          <p className="text-muted-foreground">Record food waste to improve future planning</p>
        </header>

        <Card className="p-6 space-y-6">
          <div className="space-y-3">
            <label className="text-sm font-medium text-foreground">
              Frequently Bought Items
            </label>
            <div className="flex flex-wrap gap-2">
              {isLoadingFrequent ? (
                <span className="text-xs text-muted-foreground">Loading...</span>
              ) : frequentItems.length === 0 ? (
                <span className="text-xs text-muted-foreground">No frequent items yet.</span>
              ) : (
                frequentItems.map((item) => (
                  <Button
                    key={item}
                    variant="outline"
                    size="sm"
                    onClick={() => setItemName(item)}
                    className="text-xs"
                  >
                    {item}
                  </Button>
                ))
              )}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Item Name
            </label>
            <Input
              placeholder="Enter item name"
              value={itemName}
              onChange={(e) => setItemName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Quantity Wasted
            </label>
            <div className="flex gap-3">
              <Input
                type="number"
                step="0.01"
                min="0"
                placeholder="Enter amount"
                value={quantityValue}
                onChange={(e) => setQuantityValue(e.target.value)}
              />
              <Input
                placeholder="Unit (e.g., kg, pieces)"
                value={quantityUnit}
                onChange={(e) => setQuantityUnit(e.target.value)}
              />
            </div>
          </div>

          <Button 
            onClick={handleLogWaste}
            className="w-full"
            variant="destructive"
            disabled={!itemName || !quantityValue || isSubmitting}
          >
            <Trash2 className="w-4 h-4 mr-2" />
            {isSubmitting ? "Logging..." : "Log Waste"}
          </Button>
        </Card>
      </div>
    </Layout>
  );
};

export default AddItems;
