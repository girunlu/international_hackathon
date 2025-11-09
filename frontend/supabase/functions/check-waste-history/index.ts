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
    const { itemName, requestedAmount } = await req.json();
    
    console.log('Checking waste history for:', itemName, 'Requested:', requestedAmount);
    
    // TODO: Replace this with your Python function that:
    // 1. Reads waste_log.csv
    // 2. Filters by item_name where waste_date is within last 7 days
    // 3. Gets the most recent waste entry for this item
    // 4. Calculates suggested reduced amount based on waste history
    // 5. Returns hasWaste: true/false, wastedAmount, and suggestedAmount
    
    // Mock response - replace with actual CSV reading logic
    const mockWasteData: { [key: string]: { amount: string, suggestedAmount: string } } = {
      'Milk': { amount: '0.5L', suggestedAmount: '1.5L' },
      'Bread': { amount: '2 pieces', suggestedAmount: '1 unit' },
      'Lettuce': { amount: '0.3kg', suggestedAmount: '0.5kg' },
    };
    
    const wasteInfo = mockWasteData[itemName];
    
    if (wasteInfo) {
      return new Response(
        JSON.stringify({ 
          hasWaste: true, 
          wastedAmount: wasteInfo.amount,
          suggestedAmount: wasteInfo.suggestedAmount,
          itemName 
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }
    
    return new Response(
      JSON.stringify({ hasWaste: false }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
    
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
