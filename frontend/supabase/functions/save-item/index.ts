import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { itemName, originalAmount, savedAmount } = await req.json();
    
    console.log('Saving item:', { itemName, originalAmount, savedAmount });
    
    // TODO: Replace this with your Python function that:
    // 1. Reads saved_items.csv (or creates if doesn't exist)
    // 2. Appends new row with: id, item_name, date, original_amount, saved_amount
    // 3. Calculates savings based on difference
    
    // Mock response - replace with actual CSV writing logic
    const savedItem = {
      id: Date.now(),
      item_name: itemName,
      date: new Date().toISOString().split('T')[0],
      original_amount: originalAmount,
      saved_amount: savedAmount,
      saved_by: calculateSavings(originalAmount, savedAmount)
    };
    
    console.log('Saved item to CSV:', savedItem);
    
    return new Response(
      JSON.stringify({ 
        success: true, 
        savedItem 
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
    
  } catch (error) {
    console.error('Error in save-item:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      { 
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});

function calculateSavings(original: string, saved: string): string {
  // Simple calculation - you can enhance this
  const origNum = parseFloat(original);
  const savedNum = parseFloat(saved);
  if (!isNaN(origNum) && !isNaN(savedNum)) {
    const diff = origNum - savedNum;
    return `${diff}${original.replace(/[\d.]/g, '')}`;
  }
  return '0';
}
