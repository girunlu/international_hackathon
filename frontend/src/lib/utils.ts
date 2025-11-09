import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function parseQuantityToFloat(quantity: string): number {
  if (!quantity) {
    return 0;
  }

  const normalized = String(quantity).replace(",", ".").trim();
  const match = normalized.match(/-?\d+(\.\d+)?/);

  if (!match) {
    return 0;
  }

  const value = parseFloat(match[0]);
  return Number.isNaN(value) ? 0 : value;
}

export function parseAmountToFloat(amount: string | number): number {
  if (typeof amount === "number") {
    return Number.isNaN(amount) ? 0 : amount;
  }

  if (typeof amount === "string") {
    const cleaned = amount.replace(/[^\d.,-]/g, "").replace(",", ".").trim();
    if (!cleaned) {
      return 0;
    }

    const value = parseFloat(cleaned);
    return Number.isNaN(value) ? 0 : value;
  }

  return 0;
}

export function formatQuantityLabel(quantityNumeric: number, unit?: string | null): string {
  if (!Number.isFinite(quantityNumeric)) {
    return "0";
  }

  const normalized = Number(quantityNumeric);
  const numericString = Number.isInteger(normalized)
    ? normalized.toString()
    : normalized.toFixed(2).replace(/\.?0+$/, "");

  if (!unit) {
    return numericString;
  }

  return `${numericString} ${unit}`;
}
