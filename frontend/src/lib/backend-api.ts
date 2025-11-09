/**
 * Backend API Service
 * Handles all communication with the Python backend server running on localhost:8000
 */

import { formatQuantityLabel } from "./utils";

const BACKEND_URL = "http://127.0.0.1:8000";
const STATIC_BASE_URL = `${(import.meta.env.BASE_URL ?? "/").replace(/\/$/, "")}/static_files`;

const STATIC_ENDPOINT_MAP: Record<string, string> = {
  "/get-shopping-list": "shopping_list.json",
  "/get-shopping-history": "shopping_history.json",
  "/get-frequent-items": "frequent_items.json",
  "/analyze-waste": "analytics_summary.json",
  "/monthly-analytics-summary": "analytics_summary.json",
  "/weekly-recommendations": "weekly_recommendations.json",
};

async function fetchStaticJson<T = any>(filename: string): Promise<T> {
  const response = await fetch(`${STATIC_BASE_URL}/${filename}`, {
    method: "GET",
    cache: "no-cache",
  });

  if (!response.ok) {
    throw new Error(`Static file error: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

async function fallbackToStatic(endpoint: string): Promise<any> {
  const staticFile = STATIC_ENDPOINT_MAP[endpoint];
  if (!staticFile) {
    throw new Error(`No static fallback defined for ${endpoint}`);
  }
  return fetchStaticJson(staticFile);
}

/**
 * Generic function to call Python backend endpoints
 */
async function callBackend(endpoint: string, data: any): Promise<any> {
  const staticFile = STATIC_ENDPOINT_MAP[endpoint];

  try {
    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      if (staticFile) {
        console.warn(`Backend returned ${response.status} for ${endpoint}, falling back to static file.`);
        return fallbackToStatic(endpoint);
      }

      throw new Error(`Backend error: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    if (staticFile) {
      console.warn(`Backend unavailable for ${endpoint}, using static file data.`, error);
      return fallbackToStatic(endpoint);
    }

    console.error(`Error calling backend endpoint ${endpoint}:`, error);
    throw error;
  }
}

/**
 * Detect item from image (base64 encoded)
 */
export async function detectItem(image: string): Promise<{ item_name: string }> {
  return callBackend("/detect-item", { image });
}

/**
 * Check waste history for an item
 */
export interface WasteHistoryResponse {
  hasWaste: boolean;
  itemName?: string;
  wastedAmount?: string;
  wastedQuantityNumeric?: number;
  suggestedAmount?: string;
  suggestedQuantityNumeric?: number;
  suggestedUnit?: string | null;
}

export async function checkWasteHistory(
  itemName: string,
  quantityNumeric: number,
  unit?: string | null
): Promise<WasteHistoryResponse> {
  const payload = {
    itemName,
    quantity_numeric: parseFloat(quantityNumeric.toFixed(4)),
    unit: unit ?? undefined,
  };

  return callBackend("/check-waste-history", payload);
}

/**
 * Analyze waste data with filters
 */
export async function analyzeWaste(
  filter: string,
  timeRange: string
): Promise<{
  wastePercentage: number;
  totalSpent: number;
  totalSaved: number;
  totalWasted?: number;
  topWasted: Array<{ name: string; count: number; amount: number }>;
  topSaved: Array<{ name: string; count: number; amount: number }>;
}> {
  return callBackend("/analyze-waste", { filter, timeRange });
}

/**
 * Get weekly recommendations
 */
export async function getWeeklyRecommendations(): Promise<{
  recommendation: string;
}> {
  return callBackend("/weekly-recommendations", {});
}

export interface AnalyticsSeriesPoint {
  date: string;
  spent: number;
  saved: number;
  wasted: number;
  rangeStart?: string;
  rangeEnd?: string;
}

export interface AnalyticsSeriesResponse {
  timeRange: "weekly" | "monthly";
  itemName: string;
  series: AnalyticsSeriesPoint[];
  totals?: {
    spent?: number;
    saved?: number;
    wasted?: number;
    wastePercentage?: number;
  };
  availableItems?: string[];
}

export async function fetchAnalyticsSeries(
  timeRange: "weekly" | "monthly",
  itemName?: string,
): Promise<AnalyticsSeriesResponse> {
  return callBackend("/analytics", {
    timeRange,
    itemName,
  });
}

export async function getAnalyticsSummary(): Promise<{
  wastePercentage: number;
  totalSpent: number;
  totalSaved: number;
  totalWasted: number;
  topWasted: Array<{ name: string; count: number; amount: number }>;
  topSaved: Array<{ name: string; amount: number; count: number }>;
}> {
  return callBackend("/monthly-analytics-summary", {});
}

/**
 * Log waste entry
 */
export async function logWaste(
  itemName: string,
  quantityNumeric: number,
  unit: string | null,
  price?: number | null
): Promise<{
  success: boolean;
}> {
  const quantityFloat = parseFloat(quantityNumeric.toFixed(4));
  const formattedQuantity = formatQuantityLabel(quantityFloat, unit ?? undefined);
  const normalizedPrice =
    typeof price === "number" && Number.isFinite(price) ? parseFloat(price.toFixed(2)) : undefined;

  return callBackend("/log-waste", {
    itemName,
    quantity: formattedQuantity,
    quantity_numeric: quantityFloat,
    unit: unit ?? undefined,
    price: normalizedPrice,
  });
}

/**
 * Mark items as bought (move from shopping list to purchases)
 */
export async function markItemsAsBought(items: Array<{
  item_name: string;
  quantity_numeric: number;
  unit?: string | null;
}>): Promise<{
  success: boolean;
}> {
  // Parse quantities to floats for better backend processing
  const itemsWithNumeric = items.map((item) => ({
    item_name: item.item_name,
    quantity: formatQuantityLabel(item.quantity_numeric, item.unit ?? undefined),
    quantity_numeric: parseFloat(item.quantity_numeric.toFixed(4)),
    unit: item.unit ?? undefined,
  }));

  return callBackend("/mark-items-bought", { items: itemsWithNumeric });
}

/**
 * Get shopping history
 */
export async function getShoppingHistory(): Promise<
  Array<{
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
  }>
> {
  await callBackend("/get-shopping-history", {});

  try {
    const response = await fetch(`${STATIC_BASE_URL}/shopping_history.json`, {
      method: "GET",
      cache: "no-cache",
    });
    if (!response.ok) {
      throw new Error(`Unable to load shopping_history.json: ${response.statusText}`);
    }
    const data = (await response.json()) as { shoppingTrips?: any };
    if (Array.isArray(data)) {
      return data as any;
    }
    return Array.isArray(data?.shoppingTrips) ? data.shoppingTrips : [];
  } catch (error) {
    console.error("Failed to read shopping history from static file:", error);
    return [];
  }
}

/**
 * Get frequent items (for AddItems page)
 */
export async function getFrequentItems(): Promise<string[]> {
  await callBackend("/get-frequent-items", {});

  try {
    const response = await fetch(`${STATIC_BASE_URL}/frequent_items.json`, {
      method: "GET",
      cache: "no-cache",
    });
    if (!response.ok) {
      throw new Error(`Unable to load frequent_items.json: ${response.statusText}`);
    }
    const data = (await response.json()) as { items?: string[] };
    if (Array.isArray(data)) {
      return data as string[];
    }
    return Array.isArray(data?.items) ? data.items : [];
  } catch (error) {
    console.error("Failed to read frequent items from static file:", error);
    return [];
  }
}

export interface ShoppingListEntry {
  id: number;
  item_name: string;
  quantity_numeric: number;
  quantity_unit?: string | null;
  date_added: string;
}

function normalizeShoppingListEntry(entry: any): ShoppingListEntry | null {
  if (!entry || typeof entry !== "object") {
    return null;
  }

  const id = typeof entry.id === "number" ? entry.id : Number(entry.id);
  const itemName = typeof entry.item_name === "string" ? entry.item_name : entry.itemName;
  const quantityNumeric =
    typeof entry.quantity_numeric === "number"
      ? entry.quantity_numeric
      : typeof entry.quantity === "number"
      ? entry.quantity
      : entry.quantity
      ? parseFloat(String(entry.quantity))
      : NaN;
  const quantityUnit =
    typeof entry.quantity_unit === "string"
      ? entry.quantity_unit
      : typeof entry.unit === "string"
      ? entry.unit
      : undefined;
  const dateAdded =
    typeof entry.date_added === "string"
      ? entry.date_added
      : typeof entry.dateAdded === "string"
      ? entry.dateAdded
      : new Date().toISOString().split("T")[0];

  if (!Number.isFinite(id) || !itemName) {
    return null;
  }

  return {
    id,
    item_name: itemName,
    quantity_numeric: Number.isFinite(quantityNumeric) ? quantityNumeric : 0,
    quantity_unit: quantityUnit,
    date_added: dateAdded,
  };
}

export async function getShoppingList(): Promise<ShoppingListEntry[]> {
  await callBackend("/get-shopping-list", {});

  try {
    const response = await fetch(`${STATIC_BASE_URL}/shopping_list.json`, {
      method: "GET",
      cache: "no-cache",
    });
    if (!response.ok) {
      throw new Error(`Unable to load shopping_list.json: ${response.statusText}`);
    }
    const data = (await response.json()) as any[];
    return data
      .map(normalizeShoppingListEntry)
      .filter((entry): entry is ShoppingListEntry => entry !== null);
  } catch (error) {
    console.error("Failed to read shopping list from static file:", error);
    return [];
  }
}

export async function addShoppingListItem(
  itemName: string,
  quantityNumeric: number,
  unit?: string | null
): Promise<ShoppingListEntry> {
  const quantityFloat = parseFloat(quantityNumeric.toFixed(4));
  const payload = {
    itemName,
    quantity: formatQuantityLabel(quantityFloat, unit ?? undefined),
    quantity_numeric: quantityFloat,
    unit: unit ?? undefined,
  };

  const result = await callBackend("/add-shopping-item", payload);

  const directItem = normalizeShoppingListEntry(result?.item);
  if (result?.success && directItem) {
    return directItem;
  }

  await callBackend("/get-shopping-list", {});

  const response = await fetch(`${STATIC_BASE_URL}/shopping_list.json`, {
    method: "GET",
    cache: "no-cache",
  });
  if (!response.ok) {
    throw new Error(`Unable to load shopping_list.json: ${response.statusText}`);
  }

  const data = (await response.json()) as any[];
  const normalized = data
    .map(normalizeShoppingListEntry)
    .filter((entry): entry is ShoppingListEntry => entry !== null);

  if (normalized.length === 0) {
    throw new Error("Shopping list snapshot empty after add");
  }

  return normalized[normalized.length - 1];
}

export async function removeShoppingListItem(id: number): Promise<{ success: boolean }> {
  const result = await callBackend("/remove-shopping-item", { id });

  try {
    const response = await fetch(`${STATIC_BASE_URL}/shopping_list.json`, {
      method: "GET",
      cache: "no-cache",
    });
    if (!response.ok) {
      throw new Error(`Unable to load shopping_list.json: ${response.statusText}`);
    }
    return { success: !!result?.success };
  } catch (error) {
    console.error("Failed to refresh shopping list after removal:", error);
    return { success: false };
  }
}

export async function recordSavedItem(options: {
  itemName: string;
  originalQuantity: number;
  originalUnit?: string | null;
  savedQuantity: number;
  savedUnit?: string | null;
}): Promise<ShoppingListEntry> {
  const savedQuantityFloat = parseFloat(options.savedQuantity.toFixed(4));
  const payload = {
    itemName: options.itemName,
    originalQuantity: parseFloat(options.originalQuantity.toFixed(4)),
    originalUnit: options.originalUnit ?? undefined,
    finalQuantity: savedQuantityFloat,
    savedQuantity: savedQuantityFloat,
    savedUnit: options.savedUnit ?? undefined,
    unit: options.savedUnit ?? undefined,
  };

  const result = await callBackend("/record-saved-item", payload);
  const directItem = normalizeShoppingListEntry(result?.item);

  if (result?.success && directItem) {
    return directItem;
  }

  await callBackend("/get-shopping-list", {});

  const response = await fetch(`${STATIC_BASE_URL}/shopping_list.json`, {
    method: "GET",
    cache: "no-cache",
  });
  if (!response.ok) {
    throw new Error(`Unable to load shopping_list.json: ${response.statusText}`);
  }

  const data = (await response.json()) as any[];
  const normalized = data
    .map(normalizeShoppingListEntry)
    .filter((entry): entry is ShoppingListEntry => entry !== null);

  if (normalized.length === 0) {
    throw new Error("Shopping list snapshot empty after saved item record");
  }

  return normalized[normalized.length - 1];
}

