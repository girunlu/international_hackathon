import "https://deno.land/x/xhr@0.1.0/mod.ts";
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
    const { filter, timeRange } = await req.json();

    const payload = {
      filter: typeof filter === "string" && filter.trim() ? filter.trim() : "general",
      timeRange: typeof timeRange === "string" && timeRange.trim() ? timeRange.trim() : "weekly",
    };

    const result = await callBackend("/analyze-waste", payload);

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error in analyze-waste:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
