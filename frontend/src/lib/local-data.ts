/**
 * Local Data Service
 * Manages local data storage (using localStorage as a simple solution)
 * Can be replaced with CSV file reading if needed
 */

import { formatQuantityLabel, parseQuantityToFloat } from "./utils";

export interface ShoppingListItem {
  id: number;
  item_name: string;
  quantity_numeric: number;
  quantity_unit?: string | null;
  date_added: string;
}

export interface PurchaseItem {
  id: number;
  item_name: string;
  shopping_date: string;
  quantity: string;
  price: number;
}

export interface WasteLogItem {
  id: number;
  item_name: string;
  quantity: string;
  price: number;
  waste_date: string;
}

const STORAGE_KEYS = {
  SHOPPING_LIST: "shopping_list",
  PURCHASES: "purchases",
  WASTE_LOG: "waste_log",
} as const;

/**
 * Initialize local storage with empty arrays if they don't exist
 */
function initializeStorage() {
  if (!localStorage.getItem(STORAGE_KEYS.SHOPPING_LIST)) {
    localStorage.setItem(STORAGE_KEYS.SHOPPING_LIST, JSON.stringify([]));
  }
  if (!localStorage.getItem(STORAGE_KEYS.PURCHASES)) {
    localStorage.setItem(STORAGE_KEYS.PURCHASES, JSON.stringify([]));
  }
  if (!localStorage.getItem(STORAGE_KEYS.WASTE_LOG)) {
    localStorage.setItem(STORAGE_KEYS.WASTE_LOG, JSON.stringify([]));
  }
}

// Initialize on module load
initializeStorage();

/**
 * Shopping List Operations
 */
export const shoppingListService = {
  getAll(): ShoppingListItem[] {
    const data = localStorage.getItem(STORAGE_KEYS.SHOPPING_LIST);
    if (!data) {
      return [];
    }

    try {
      const parsed: unknown = JSON.parse(data);
      if (!Array.isArray(parsed)) {
        return [];
      }

      return parsed
        .map((entry): ShoppingListItem | null => {
          if (typeof entry !== "object" || entry === null) {
            return null;
          }

          const item = entry as Record<string, unknown>;
          const id = typeof item.id === "number" ? item.id : Date.now();
          const itemName =
            typeof item.item_name === "string"
              ? item.item_name
              : typeof item.name === "string"
              ? item.name
              : "";

          if (!itemName) {
            return null;
          }

          let quantityNumeric: number;
          if (typeof item.quantity_numeric === "number") {
            quantityNumeric = item.quantity_numeric;
          } else if (typeof item.quantity === "string") {
            quantityNumeric = parseQuantityToFloat(item.quantity);
          } else {
            quantityNumeric = 0;
          }

          const quantityUnit =
            typeof item.quantity_unit === "string" && item.quantity_unit.trim().length > 0
              ? item.quantity_unit.trim()
              : undefined;

          const dateAdded =
            typeof item.date_added === "string"
              ? item.date_added
              : new Date().toISOString().split("T")[0];

          return {
            id,
            item_name: itemName,
            quantity_numeric: Number.isFinite(quantityNumeric) ? quantityNumeric : 0,
            quantity_unit: quantityUnit,
            date_added: dateAdded,
          };
        })
        .filter((item): item is ShoppingListItem => item !== null);
    } catch (error) {
      console.error("Failed to parse shopping list storage", error);
      return [];
    }
  },

  add(item: Omit<ShoppingListItem, "id">): ShoppingListItem {
    const items = this.getAll();
    const newItem: ShoppingListItem = {
      id: Date.now(),
      ...item,
      quantity_numeric: parseFloat(item.quantity_numeric.toFixed(4)),
      quantity_unit: item.quantity_unit?.trim() || undefined,
    };
    items.push(newItem);
    localStorage.setItem(STORAGE_KEYS.SHOPPING_LIST, JSON.stringify(items));
    return newItem;
  },

  remove(id: number): void {
    const items = this.getAll().filter((item) => item.id !== id);
    localStorage.setItem(STORAGE_KEYS.SHOPPING_LIST, JSON.stringify(items));
  },

  clear(): void {
    localStorage.setItem(STORAGE_KEYS.SHOPPING_LIST, JSON.stringify([]));
  },
};

/**
 * Purchases Operations
 */
export const purchasesService = {
  getAll(): PurchaseItem[] {
    const data = localStorage.getItem(STORAGE_KEYS.PURCHASES);
    return data ? JSON.parse(data) : [];
  },

  add(item: Omit<PurchaseItem, "id">): PurchaseItem {
    const items = this.getAll();
    const newItem: PurchaseItem = {
      id: Date.now(),
      ...item,
    };
    items.push(newItem);
    localStorage.setItem(STORAGE_KEYS.PURCHASES, JSON.stringify(items));
    return newItem;
  },

  addMultiple(items: Omit<PurchaseItem, "id">[]): PurchaseItem[] {
    const existingItems = this.getAll();
    const newItems: PurchaseItem[] = items.map((item) => ({
      id: Date.now() + Math.random(),
      ...item,
    }));
    const allItems = [...existingItems, ...newItems];
    localStorage.setItem(STORAGE_KEYS.PURCHASES, JSON.stringify(allItems));
    return newItems;
  },
};

/**
 * Waste Log Operations
 */
export const wasteLogService = {
  getAll(): WasteLogItem[] {
    const data = localStorage.getItem(STORAGE_KEYS.WASTE_LOG);
    return data ? JSON.parse(data) : [];
  },

  add(item: Omit<WasteLogItem, "id">): WasteLogItem {
    const items = this.getAll();
    const newItem: WasteLogItem = {
      id: Date.now(),
      ...item,
    };
    items.push(newItem);
    localStorage.setItem(STORAGE_KEYS.WASTE_LOG, JSON.stringify(items));
    return newItem;
  },
};

/**
 * Export data as CSV format (for debugging or backup)
 */
export function exportDataAsCSV(): {
  shopping_list: string;
  purchases: string;
  waste_log: string;
} {
  const shoppingList = shoppingListService.getAll();
  const purchases = purchasesService.getAll();
  const wasteLog = wasteLogService.getAll();

  const shoppingListCSV = [
    "id,item_name,quantity_numeric,quantity_unit,quantity_label,date_added",
    ...shoppingList.map(
      (item) =>
        `${item.id},${item.item_name},${item.quantity_numeric},${item.quantity_unit ?? ""},${formatQuantityLabel(
          item.quantity_numeric,
          item.quantity_unit
        )},${item.date_added}`
    ),
  ].join("\n");

  const purchasesCSV = [
    "id,item_name,shopping_date,quantity,price",
    ...purchases.map(
      (item) =>
        `${item.id},${item.item_name},${item.shopping_date},${item.quantity},${item.price}`
    ),
  ].join("\n");

  const wasteLogCSV = [
    "id,item_name,quantity,price,waste_date",
    ...wasteLog.map(
      (item) =>
        `${item.id},${item.item_name},${item.quantity},${item.price},${item.waste_date}`
    ),
  ].join("\n");

  return {
    shopping_list: shoppingListCSV,
    purchases: purchasesCSV,
    waste_log: wasteLogCSV,
  };
}

