"""
Broad Market Universes across US Equities, Sectors, Crypto, Forex, and Commodities.
"""

from dataclasses import dataclass
from typing import List
from trading_bot.core.instruments import AssetClass


@dataclass
class MarketAssetInfo:
    symbol: str
    name: str
    asset_class: AssetClass
    sector: str
    tick_size: float = 0.01
    lot_size: float = 1.0
    fee_rate: float = 0.0005


# Comprehensive Multi-Asset Market Universe (62 assets across all sectors)
MARKET_UNIVERSE: List[MarketAssetInfo] = [
    # --- Broad Market & Sector ETFs ---
    MarketAssetInfo("SPY", "S&P 500 ETF", AssetClass.STOCK, "Broad Market", 0.01, 1.0, 0.0002),
    MarketAssetInfo("QQQ", "Nasdaq 100 ETF", AssetClass.STOCK, "Broad Market", 0.01, 1.0, 0.0002),
    MarketAssetInfo("IWM", "Russell 2000 Small Cap ETF", AssetClass.STOCK, "Broad Market", 0.01, 1.0, 0.0003),
    MarketAssetInfo("DIA", "Dow Jones Industrial ETF", AssetClass.STOCK, "Broad Market", 0.01, 1.0, 0.0002),
    MarketAssetInfo("XLK", "Technology Select SPDR", AssetClass.STOCK, "Technology", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLF", "Financial Select SPDR", AssetClass.STOCK, "Financials", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLE", "Energy Select SPDR", AssetClass.STOCK, "Energy", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLV", "Health Care Select SPDR", AssetClass.STOCK, "Healthcare", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLY", "Consumer Discretionary SPDR", AssetClass.STOCK, "Consumer", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLP", "Consumer Staples SPDR", AssetClass.STOCK, "Consumer", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLI", "Industrials Select SPDR", AssetClass.STOCK, "Industrials", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLU", "Utilities Select SPDR", AssetClass.STOCK, "Utilities", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLC", "Communication Services SPDR", AssetClass.STOCK, "Communication", 0.01, 1.0, 0.0003),
    MarketAssetInfo("XLB", "Materials Select SPDR", AssetClass.STOCK, "Materials", 0.01, 1.0, 0.0003),
    MarketAssetInfo("TLT", "20+ Year Treasury Bond ETF", AssetClass.STOCK, "Fixed Income", 0.01, 1.0, 0.0002),

    # --- Mega-Cap & Sector Equities ---
    MarketAssetInfo("AAPL", "Apple Inc.", AssetClass.STOCK, "Technology", 0.01, 1.0, 0.0005),
    MarketAssetInfo("MSFT", "Microsoft Corp.", AssetClass.STOCK, "Technology", 0.01, 1.0, 0.0005),
    MarketAssetInfo("NVDA", "NVIDIA Corp.", AssetClass.STOCK, "Semiconductors", 0.01, 1.0, 0.0005),
    MarketAssetInfo("GOOGL", "Alphabet Inc.", AssetClass.STOCK, "Communication", 0.01, 1.0, 0.0005),
    MarketAssetInfo("AMZN", "Amazon.com Inc.", AssetClass.STOCK, "Consumer", 0.01, 1.0, 0.0005),
    MarketAssetInfo("META", "Meta Platforms Inc.", AssetClass.STOCK, "Communication", 0.01, 1.0, 0.0005),
    MarketAssetInfo("TSLA", "Tesla Inc.", AssetClass.STOCK, "Automotive/Tech", 0.01, 1.0, 0.0005),
    MarketAssetInfo("AMD", "Advanced Micro Devices", AssetClass.STOCK, "Semiconductors", 0.01, 1.0, 0.0005),
    MarketAssetInfo("AVGO", "Broadcom Inc.", AssetClass.STOCK, "Semiconductors", 0.01, 1.0, 0.0005),
    MarketAssetInfo("NFLX", "Netflix Inc.", AssetClass.STOCK, "Communication", 0.01, 1.0, 0.0005),
    MarketAssetInfo("JPM", "JPMorgan Chase & Co.", AssetClass.STOCK, "Financials", 0.01, 1.0, 0.0005),
    MarketAssetInfo("BAC", "Bank of America Corp.", AssetClass.STOCK, "Financials", 0.01, 1.0, 0.0005),
    MarketAssetInfo("V", "Visa Inc.", AssetClass.STOCK, "Financials", 0.01, 1.0, 0.0005),
    MarketAssetInfo("MA", "Mastercard Inc.", AssetClass.STOCK, "Financials", 0.01, 1.0, 0.0005),
    MarketAssetInfo("UNH", "UnitedHealth Group Inc.", AssetClass.STOCK, "Healthcare", 0.01, 1.0, 0.0005),
    MarketAssetInfo("JNJ", "Johnson & Johnson", AssetClass.STOCK, "Healthcare", 0.01, 1.0, 0.0005),
    MarketAssetInfo("LLY", "Eli Lilly and Co.", AssetClass.STOCK, "Healthcare", 0.01, 1.0, 0.0005),
    MarketAssetInfo("PFE", "Pfizer Inc.", AssetClass.STOCK, "Healthcare", 0.01, 1.0, 0.0005),
    MarketAssetInfo("XOM", "Exxon Mobil Corp.", AssetClass.STOCK, "Energy", 0.01, 1.0, 0.0005),
    MarketAssetInfo("CVX", "Chevron Corp.", AssetClass.STOCK, "Energy", 0.01, 1.0, 0.0005),
    MarketAssetInfo("WMT", "Walmart Inc.", AssetClass.STOCK, "Consumer", 0.01, 1.0, 0.0005),
    MarketAssetInfo("COST", "Costco Wholesale Corp.", AssetClass.STOCK, "Consumer", 0.01, 1.0, 0.0005),
    MarketAssetInfo("HD", "Home Depot Inc.", AssetClass.STOCK, "Consumer", 0.01, 1.0, 0.0005),
    MarketAssetInfo("DIS", "Walt Disney Co.", AssetClass.STOCK, "Entertainment", 0.01, 1.0, 0.0005),
    MarketAssetInfo("PLTR", "Palantir Technologies", AssetClass.STOCK, "Technology", 0.01, 1.0, 0.0005),

    # --- Crypto Market Universe ---
    MarketAssetInfo("BTC-USD", "Bitcoin", AssetClass.CRYPTO_SPOT, "Crypto L1", 0.1, 0.001, 0.0006),
    MarketAssetInfo("ETH-USD", "Ethereum", AssetClass.CRYPTO_SPOT, "Crypto L1", 0.01, 0.01, 0.0006),
    MarketAssetInfo("SOL-USD", "Solana", AssetClass.CRYPTO_SPOT, "Crypto L1", 0.01, 0.1, 0.0006),
    MarketAssetInfo("BNB-USD", "Binance Coin", AssetClass.CRYPTO_SPOT, "Crypto L1", 0.01, 0.1, 0.0006),
    MarketAssetInfo("XRP-USD", "Ripple", AssetClass.CRYPTO_SPOT, "Crypto Payments", 0.0001, 1.0, 0.0006),
    MarketAssetInfo("ADA-USD", "Cardano", AssetClass.CRYPTO_SPOT, "Crypto L1", 0.0001, 1.0, 0.0006),
    MarketAssetInfo("DOGE-USD", "Dogecoin", AssetClass.CRYPTO_SPOT, "Crypto Meme", 0.00001, 10.0, 0.0006),
    MarketAssetInfo("AVAX-USD", "Avalanche", AssetClass.CRYPTO_SPOT, "Crypto L1", 0.01, 0.1, 0.0006),
    MarketAssetInfo("LINK-USD", "Chainlink", AssetClass.CRYPTO_SPOT, "Crypto Oracle", 0.001, 0.1, 0.0006),
    MarketAssetInfo("DOT-USD", "Polkadot", AssetClass.CRYPTO_SPOT, "Crypto L1", 0.001, 0.1, 0.0006),

    # --- Forex Currency Pairs ---
    MarketAssetInfo("EURUSD=X", "EUR/USD", AssetClass.FOREX, "Forex Majors", 0.0001, 1000.0, 0.00005),
    MarketAssetInfo("GBPUSD=X", "GBP/USD", AssetClass.FOREX, "Forex Majors", 0.0001, 1000.0, 0.00005),
    MarketAssetInfo("USDJPY=X", "USD/JPY", AssetClass.FOREX, "Forex Majors", 0.01, 1000.0, 0.00005),
    MarketAssetInfo("AUDUSD=X", "AUD/USD", AssetClass.FOREX, "Forex Majors", 0.0001, 1000.0, 0.00005),
    MarketAssetInfo("USDCAD=X", "USD/CAD", AssetClass.FOREX, "Forex Majors", 0.0001, 1000.0, 0.00005),
    MarketAssetInfo("USDCHF=X", "USD/CHF", AssetClass.FOREX, "Forex Majors", 0.0001, 1000.0, 0.00005),
    MarketAssetInfo("EURGBP=X", "EUR/GBP", AssetClass.FOREX, "Forex Crosses", 0.0001, 1000.0, 0.00005),
    MarketAssetInfo("EURJPY=X", "EUR/JPY", AssetClass.FOREX, "Forex Crosses", 0.01, 1000.0, 0.00005),

    # --- Commodities & Credit ---
    MarketAssetInfo("GLD", "SPDR Gold Shares", AssetClass.COMMODITY, "Commodities", 0.01, 1.0, 0.0003),
    MarketAssetInfo("SLV", "iShares Silver Trust", AssetClass.COMMODITY, "Commodities", 0.01, 1.0, 0.0003),
    MarketAssetInfo("USO", "United States Oil Fund", AssetClass.COMMODITY, "Commodities", 0.01, 1.0, 0.0005),
    MarketAssetInfo("HYG", "iShares High Yield Corporate Bond", AssetClass.STOCK, "Fixed Income", 0.01, 1.0, 0.0003),
]

