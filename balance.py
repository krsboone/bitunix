"""
balance.py — Check Bitunix futures account balance

Usage:
    python3 balance.py
"""

from auth import BitunixClient
from config import API_KEY, SECRET_KEY


def check_balance(client: BitunixClient, margin_coin: str = "USDT") -> None:
    print(f"\nBitunix Futures — Account Balance ({margin_coin})")
    print("─" * 50)

    result = client.get("/api/v1/futures/account", {"marginCoin": margin_coin})

    if result.get("code") != 0:
        print(f"  Error {result.get('code')}: {result.get('msg')}")
        return

    data = result.get("data", {})

    available  = float(data.get("available",             0))
    frozen     = float(data.get("frozen",                0))
    margin     = float(data.get("margin",                0))
    cross_upnl = float(data.get("crossUnrealizedPNL",    0))
    iso_upnl   = float(data.get("isolationUnrealizedPNL",0))
    bonus      = float(data.get("bonus",                 0))
    pos_mode   = data.get("positionMode", "unknown")

    # True buying power for cross-margin includes unrealized PnL
    buying_power = available + cross_upnl

    print(f"  Available       : {available:>12.4f} {margin_coin}")
    print(f"  Frozen (orders) : {frozen:>12.4f} {margin_coin}")
    print(f"  In positions    : {margin:>12.4f} {margin_coin}")
    print(f"  Cross uPnL      : {cross_upnl:>+12.4f} {margin_coin}")
    print(f"  Isolated uPnL   : {iso_upnl:>+12.4f} {margin_coin}")
    print(f"  Buying power    : {buying_power:>12.4f} {margin_coin}")
    if bonus > 0:
        print(f"  Bonus           : {bonus:>12.4f} {margin_coin}  (non-withdrawable)")
    print(f"  Position mode   : {pos_mode}")
    print()


if __name__ == "__main__":
    client = BitunixClient(API_KEY, SECRET_KEY)
    check_balance(client)
