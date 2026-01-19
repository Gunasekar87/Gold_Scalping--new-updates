
import MetaTrader5 as mt5
import sys

def main():
    if not mt5.initialize():
        print(f"MT5 Init Failed: {mt5.last_error()}")
        return

    print("MT5 Initialized")
    
    # Check for XAUUSD variations
    search_terms = ["XAUUSD", "xauusd", "GOLD", "Gold"]
    found_any = False
    
    for term in search_terms:
        symbol_info = mt5.symbol_info(term)
        if symbol_info:
            print(f"Found symbol: '{symbol_info.name}' (Visible: {symbol_info.visible})")
            found_any = True
            
            # Check positions with this symbol
            positions = mt5.positions_get(symbol=symbol_info.name)
            if positions is not None:
                print(f"  Positions for '{symbol_info.name}': {len(positions)}")
                for p in positions:
                    print(f"    Ticket: {p.ticket}, Type: {p.type}, Vol: {p.volume}, Price: {p.price_open}")
            else:
                print(f"  positions_get returned None for '{symbol_info.name}'")

    if not found_any:
        print("No XAUUSD variants found via symbol_info.")
        
    # List all positions to see what symbols they have
    all_positions = mt5.positions_get()
    if all_positions:
        print(f"\nTotal Open Positions: {len(all_positions)}")
        for p in all_positions:
            print(f"  Symbol: '{p.symbol}', Type: {p.type}, Vol: {p.volume}")
    else:
        print("\nNo open positions found globally.")

    mt5.shutdown()

if __name__ == "__main__":
    main()
