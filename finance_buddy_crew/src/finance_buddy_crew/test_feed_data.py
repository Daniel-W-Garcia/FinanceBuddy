import json
# Test the tool's underlying function
from tools.extraction_tools import get_prices

# Access the tool's run method
result = get_prices.run(ticker="GME")
print("Tool result:", result)
if result.get('items'):
    print("Latest price:", result['items'][0]['close'])

with open("GME_prices.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print("Saved full result to GME_prices.json")