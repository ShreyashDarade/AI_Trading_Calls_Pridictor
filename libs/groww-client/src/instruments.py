"""
Groww Instruments Download and Parsing
Downloads and parses the official Groww instrument master CSV
"""
import asyncio
import csv
import logging
from datetime import datetime
from io import StringIO
from typing import List, Optional, Dict, Any
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


class InstrumentDownloader:
    """
    Download and parse Groww instrument master CSV
    
    The instrument CSV contains all tradeable instruments with their
    exchange tokens required for streaming subscriptions.
    
    Source: https://growwapi-assets.groww.in/instruments/instrument.csv
    
    Usage:
        downloader = InstrumentDownloader()
        instruments = await downloader.download_and_parse()
        
        # Filter by exchange
        nse_instruments = [i for i in instruments if i['exchange'] == 'NSE']
    """
    
    CSV_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"
    
    # Expected CSV columns based on Groww documentation
    EXPECTED_COLUMNS = [
        "exchange",
        "segment",
        "instrument_type",
        "trading_symbol",
        "groww_symbol",
        "exchange_token",
        "name",
        "isin",
        "lot_size",
        "tick_size",
        "expiry_date",
        "strike_price",
    ]
    
    def __init__(self, csv_url: Optional[str] = None):
        """
        Initialize downloader
        
        Args:
            csv_url: Override default CSV URL (optional)
        """
        self.csv_url = csv_url or self.CSV_URL
    
    async def download_csv(self, save_path: Optional[str] = None) -> str:
        """
        Download instrument CSV from Groww
        
        Args:
            save_path: Optional path to save CSV file
        
        Returns:
            CSV content as string
        """
        logger.info(f"Downloading instrument CSV from {self.csv_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.csv_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download CSV: {response.status}")
                
                content = await response.text()
                logger.info(f"Downloaded {len(content)} bytes")
                
                # Optionally save to file
                if save_path:
                    async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
                        await f.write(content)
                    logger.info(f"Saved CSV to {save_path}")
                
                return content
    
    def parse_csv(self, csv_content: str) -> List[Dict[str, Any]]:
        """
        Parse CSV content into list of instrument dictionaries
        
        Args:
            csv_content: Raw CSV string
        
        Returns:
            List of instrument dictionaries
        """
        instruments = []
        
        reader = csv.DictReader(StringIO(csv_content))
        
        for row in reader:
            try:
                instrument = self._parse_row(row)
                if instrument:
                    instruments.append(instrument)
            except Exception as e:
                logger.warning(f"Failed to parse row: {e}")
                continue
        
        logger.info(f"Parsed {len(instruments)} instruments")
        return instruments
    
    def _parse_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Parse a single CSV row into instrument dict"""
        # Handle different possible column names (Groww may use different casing)
        def get_value(keys: List[str], default: str = "") -> str:
            for key in keys:
                if key in row:
                    return row[key].strip()
                # Try lowercase
                if key.lower() in row:
                    return row[key.lower()].strip()
            return default
        
        exchange = get_value(["exchange", "Exchange"])
        if not exchange:
            return None
        
        trading_symbol = get_value(["trading_symbol", "tradingSymbol", "TradingSymbol"])
        if not trading_symbol:
            return None
        
        exchange_token = get_value(["exchange_token", "exchangeToken", "ExchangeToken"])
        
        # Parse expiry date if present
        expiry_str = get_value(["expiry_date", "expiryDate", "Expiry"])
        expiry_date = None
        if expiry_str:
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
            except ValueError:
                try:
                    expiry_date = datetime.strptime(expiry_str, "%d-%m-%Y")
                except ValueError:
                    pass
        
        # Parse strike price if present
        strike_str = get_value(["strike_price", "strikePrice", "Strike"])
        strike_price = None
        if strike_str:
            try:
                strike_price = float(strike_str)
            except ValueError:
                pass
        
        # Parse lot size
        lot_size_str = get_value(["lot_size", "lotSize", "LotSize"], "1")
        try:
            lot_size = int(lot_size_str)
        except ValueError:
            lot_size = 1
        
        # Parse tick size
        tick_size_str = get_value(["tick_size", "tickSize", "TickSize"], "0.05")
        try:
            tick_size = float(tick_size_str)
        except ValueError:
            tick_size = 0.05
        
        return {
            "exchange": exchange.upper(),
            "segment": get_value(["segment", "Segment"], "CASH").upper(),
            "instrument_type": get_value(["instrument_type", "instrumentType", "InstrumentType"], "EQ").upper(),
            "trading_symbol": trading_symbol,
            "groww_symbol": get_value(["groww_symbol", "growwSymbol", "GrowwSymbol"], trading_symbol),
            "exchange_token": exchange_token,
            "name": get_value(["name", "Name", "company_name"]),
            "isin": get_value(["isin", "ISIN"]),
            "lot_size": lot_size,
            "tick_size": tick_size,
            "expiry_date": expiry_date,
            "strike_price": strike_price,
            "is_tradeable": True,
        }
    
    async def download_and_parse(
        self,
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Download and parse instruments in one call
        
        Args:
            save_path: Optional path to save CSV
        
        Returns:
            List of parsed instruments
        """
        csv_content = await self.download_csv(save_path)
        return self.parse_csv(csv_content)
    
    def filter_by_exchange(
        self,
        instruments: List[Dict[str, Any]],
        exchange: str
    ) -> List[Dict[str, Any]]:
        """Filter instruments by exchange"""
        return [i for i in instruments if i["exchange"].upper() == exchange.upper()]
    
    def filter_by_segment(
        self,
        instruments: List[Dict[str, Any]],
        segment: str
    ) -> List[Dict[str, Any]]:
        """Filter instruments by segment"""
        return [i for i in instruments if i["segment"].upper() == segment.upper()]
    
    def filter_equity(
        self,
        instruments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter equity instruments only"""
        return [
            i for i in instruments
            if i["instrument_type"] == "EQ" and i["segment"] == "CASH"
        ]
    
    def search(
        self,
        instruments: List[Dict[str, Any]],
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search instruments by name or symbol
        
        Args:
            instruments: List of all instruments
            query: Search query
            limit: Maximum results
        
        Returns:
            Matching instruments
        """
        query = query.upper()
        results = []
        
        for inst in instruments:
            if (
                query in inst["trading_symbol"].upper() or
                query in inst.get("name", "").upper() or
                query in inst.get("groww_symbol", "").upper()
            ):
                results.append(inst)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_nifty50_symbols(self) -> List[str]:
        """Get list of NIFTY 50 constituent symbols"""
        # As of 2025 - update periodically
        return [
            "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
            "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
            "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
            "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
            "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
            "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
            "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
            "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
            "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM",
            "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "SHRIRAMFIN"
        ]
