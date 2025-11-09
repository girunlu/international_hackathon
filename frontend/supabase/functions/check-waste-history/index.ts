import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const BACKEND_URL = Deno.env.get("PY_BACKEND_URL") ?? "http://127.0.0.1:8000";

async function callBackend<T>(endpoint: string, payload: unknown): Promise<T> {
  const response = await fetch(`${BACKEND_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Backend ${endpoint} failed: ${response.status} ${message}`);
  }

  return response.json() as Promise<T>;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { itemName, quantity_numeric, unit } = await req.json();

    if (typeof itemName !== "string" || !itemName.trim()) {
      return new Response(
        JSON.stringify({ error: "itemName is required" }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      );
    }

    const result = await callBackend(
      "/check-waste-history",
      {
        itemName: itemName.trim(),
        quantity_numeric,
        unit,
      },
    );

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error in check-waste-history:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      { 
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});
