import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import time
import logging
import threading
import json
import hmac
import hashlib
import requests
from urllib.parse import urlencode
import websocket
import ssl

from ..base_provider import BaseDataProvider
from ..utils.rate_limiter import RateLimiter, MultiRateLimiter
from ..utils.data_normalizer import DataNormalizer

class BinanceDataProvider(BaseDataProvider):
    """
    Data provider for Binance exchange.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Binance data provider.
        
        Args:
            config: Configuration dictionary with Binance-specific settings
        """
        super().__init__(config)
        
        # Set up API endpoints
        self.base_url = config.get('base_url', 'https://api.binance.com')
        self.base_wss_url = config.get('base_wss_url', 'wss://stream.binance.com:9443')
        
        # Set up API credentials
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        
        # Set up rate limiters
        self.rate_limiter = MultiRateLimiter()
        self.rate_limiter.add_limiter('market_data', 1200, 1)  # 1200 requests per minute
        self.rate_limiter.add_limiter('order', 100, 0.1)  # 100 requests per 10 seconds
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Set up websocket connections
        self.ws_connections = {}
        self.ws_callbacks = {}
        self.ws_lock = threading.Lock()
        
        # Set up session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
    
    def connect(self) -> bool:
        """
        Establish connection to Binance.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Test connection by fetching server time
            response = self.session.get(f"{self.base_url}/api/v3/time")
            response.raise_for_status()
            self.logger.info("Connected to Binance API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance API: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Binance.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        try:
            # Close all websocket connections
            with self.ws_lock:
                for ws in self.ws_connections.values():
                    try:
                        ws.close()
                    except Exception as e:
                        self.logger.error(f"Error closing websocket: {str(e)}")
                
                self.ws_connections = {}
                self.ws_callbacks = {}
            
            # Close HTTP session
            self.session.close()
            
            self.logger.info("Disconnected from Binance API")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Binance API: {str(e)}")
            return False
    
    def get_historical_data(
        self,
        instrument: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
        interval: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical market data from Binance.
        
        Args:
            instrument: Instrument identifier (e.g., 'BTCUSDT')
            start_time: Start time for historical data
            end_time: End time for historical data
            interval: Time interval for the data (e.g., '1m', '1h', '1d')
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with historical market data
        """
        # Normalize instrument identifier
        symbol = self.normalize_instrument_id(instrument)
        
        # Convert datetime objects to milliseconds
        if isinstance(start_time, datetime):
            start_ms = int(start_time.timestamp() * 1000)
        else:
            start_ms = int(pd.to_datetime(start_time).timestamp() * 1000)
        
        if isinstance(end_time, datetime):
            end_ms = int(end_time.timestamp() * 1000)
        else:
            end_ms = int(pd.to_datetime(end_time).timestamp() * 1000)
        
        # Map interval to Binance format
        interval_map = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        
        binance_interval = interval_map.get(interval, interval)
        
        # Prepare parameters
        params = {
            'symbol': symbol,
            'interval': binance_interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1000  # Maximum allowed by Binance
        }
        
        # Fetch data in chunks if needed
        all_candles = []
        current_start = start_ms
        
        while current_start < end_ms:
            # Check rate limit
            self.rate_limiter.wait_for_token('market_data')
            
            # Make request
            try:
                response = self.session.get(f"{self.base_url}/api/v3/klines", params=params)
                response.raise_for_status()
                candles = response.json()
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update start time for next chunk
                last_candle_time = candles[-1][0]
                current_start = last_candle_time + 1
                params['startTime'] = current_start
                
                # If we got less than the limit, we've reached the end
                if len(candles) < 1000:
                    break
                
            except Exception as e:
                self.handle_error(e, "get_historical_data")
                raise
        
        # Convert to DataFrame
        if not all_candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Select only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Clean data
        df = DataNormalizer.clean_dataframe(df)
        
        return df
    
    def get_latest_data(self, instrument: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch the latest market data for a specific instrument.
        
        Args:
            instrument: Instrument identifier (e.g., 'BTCUSDT')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with latest market data
        """
        # Normalize instrument identifier
        symbol = self.normalize_instrument_id(instrument)
        
        # Check rate limit
        self.rate_limiter.wait_for_token('market_data')
        
        try:
            # Fetch ticker data
            response = self.session.get(f"{self.base_url}/api/v3/ticker/24hr", params={'symbol': symbol})
            response.raise_for_status()
            ticker_data = response.json()
            
            # Normalize data
            result = {
                'symbol': ticker_data['symbol'],
                'price': float(ticker_data['lastPrice']),
                'open': float(ticker_data['openPrice']),
                'high': float(ticker_data['highPrice']),
                'low': float(ticker_data['lowPrice']),
                'volume': float(ticker_data['volume']),
                'quote_volume': float(ticker_data['quoteVolume']),
                'change': float(ticker_data['priceChange']),
                'change_percent': float(ticker_data['priceChangePercent']),
                'trades': int(ticker_data['count']),
                'timestamp': pd.to_datetime(ticker_data['closeTime'], unit='ms')
            }
            
            return result
            
        except Exception as e:
            self.handle_error(e, "get_latest_data")
            raise
    
    def subscribe_to_stream(
        self,
        instruments: List[str],
        callback: Callable,
        stream_type: str = 'trade',
        **kwargs
    ) -> str:
        """
        Subscribe to a real-time data stream.
        
        Args:
            instruments: List of instrument identifiers
            callback: Function to call when new data is received
            stream_type: Type of stream ('trade', 'kline', 'ticker', 'depth')
            **kwargs: Additional parameters
                - interval: Required for 'kline' streams (e.g., '1m')
                - depth: Optional for 'depth' streams (default: 20)
            
        Returns:
            Stream identifier
        """
        # Normalize instrument identifiers
        symbols = [self.normalize_instrument_id(instrument).lower() for instrument in instruments]
        
        # Generate stream name based on type
        streams = []
        for symbol in symbols:
            if stream_type == 'trade':
                streams.append(f"{symbol}@trade")
            elif stream_type == 'kline':
                interval = kwargs.get('interval', '1m')
                streams.append(f"{symbol}@kline_{interval}")
            elif stream_type == 'ticker':
                streams.append(f"{symbol}@ticker")
            elif stream_type == 'depth':
                depth = kwargs.get('depth', 20)
                streams.append(f"{symbol}@depth{depth}")
            else:
                raise ValueError(f"Unsupported stream type: {stream_type}")
        
        # Generate a unique stream ID
        stream_id = f"{stream_type}_{','.join(symbols)}_{int(time.time())}"
        
        # Create websocket connection
        stream_path = "/stream?streams=" + "/".join(streams)
        ws_url = f"{self.base_wss_url}{stream_path}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                # Process data based on stream type
                if stream_type == 'trade':
                    processed_data = self._process_trade_stream(data)
                elif stream_type == 'kline':
                    processed_data = self._process_kline_stream(data)
                elif stream_type == 'ticker':
                    processed_data = self._process_ticker_stream(data)
                elif stream_type == 'depth':
                    processed_data = self._process_depth_stream(data)
                else:
                    processed_data = data
                
                # Call the callback with processed data
                callback(processed_data)
            except Exception as e:
                self.logger.error(f"Error processing websocket message: {str(e)}")
        
        def on_error(ws, error):
            self.logger.error(f"Websocket error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info(f"Websocket closed: {close_status_code} - {close_msg}")
            # Try to reconnect if this wasn't an intentional close
            with self.ws_lock:
                if stream_id in self.ws_connections and self.ws_connections[stream_id] == ws:
                    self.logger.info(f"Attempting to reconnect stream: {stream_id}")
                    self._reconnect_stream(stream_id, ws_url, on_message, on_error, on_close)
        
        def on_open(ws):
            self.logger.info(f"Websocket opened for stream: {stream_id}")
        
        # Create and start websocket
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start the websocket connection in a new thread
        wst = threading.Thread(target=ws.run_forever, kwargs={
            'sslopt': {"cert_reqs": ssl.CERT_NONE},
            'ping_interval': 30,
            'ping_timeout': 10
        })
        wst.daemon = True
        wst.start()
        
        # Store the websocket connection and callback
        with self.ws_lock:
            self.ws_connections[stream_id] = ws
            self.ws_callbacks[stream_id] = callback
        
        return stream_id
    
    def _reconnect_stream(self, stream_id, ws_url, on_message, on_error, on_close):
        """Helper method to reconnect a dropped websocket stream."""
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=lambda ws: self.logger.info(f"Reconnected stream: {stream_id}")
            )
            
            wst = threading.Thread(target=ws.run_forever, kwargs={
                'sslopt': {"cert_reqs": ssl.CERT_NONE},
                'ping_interval': 30,
                'ping_timeout': 10
            })
            wst.daemon = True
            wst.start()
            
            with self.ws_lock:
                self.ws_connections[stream_id] = ws
            
            self.logger.info(f"Successfully reconnected stream: {stream_id}")
        except Exception as e:
            self.logger.error(f"Failed to reconnect stream {stream_id}: {str(e)}")
    
    def unsubscribe_from_stream(self, stream_id: str) -> bool:
        """
        Unsubscribe from a real-time data stream.
        
        Args:
            stream_id: Stream identifier returned by subscribe_to_stream
            
        Returns:
            True if unsubscription is successful, False otherwise
        """
        with self.ws_lock:
            if stream_id in self.ws_connections:
                try:
                    ws = self.ws_connections[stream_id]
                    ws.close()
                    del self.ws_connections[stream_id]
                    
                    if stream_id in self.ws_callbacks:
                        del self.ws_callbacks[stream_id]
                    
                    self.logger.info(f"Unsubscribed from stream: {stream_id}")
                    return True
                except Exception as e:
                    self.logger.error(f"Error unsubscribing from stream {stream_id}: {str(e)}")
                    return False
            else:
                self.logger.warning(f"Stream not found: {stream_id}")
                return False
    
    def get_instruments(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Get a list of available instruments from Binance.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            List of instrument dictionaries with metadata
        """
        # Check rate limit
        self.rate_limiter.wait_for_token('market_data')
        
        try:
            # Fetch exchange information
            response = self.session.get(f"{self.base_url}/api/v3/exchangeInfo")
            response.raise_for_status()
            exchange_info = response.json()
            
            # Extract and normalize instrument information
            instruments = []
            for symbol_info in exchange_info['symbols']:
                # Skip instruments that are not trading
                if symbol_info['status'] != 'TRADING':
                    continue
                
                instrument = {
                    'symbol': symbol_info['symbol'],
                    'base_asset': symbol_info['baseAsset'],
                    'quote_asset': symbol_info['quoteAsset'],
                    'min_price': None,
                    'max_price': None,
                    'tick_size': None,
                    'min_qty': None,
                    'max_qty': None,
                    'step_size': None
                }
                
                # Extract filters
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'PRICE_FILTER':
                        instrument['min_price'] = float(filter_info['minPrice'])
                        instrument['max_price'] = float(filter_info['maxPrice'])
                        instrument['tick_size'] = float(filter_info['tickSize'])
                    elif filter_info['filterType'] == 'LOT_SIZE':
                        instrument['min_qty'] = float(filter_info['minQty'])
                        instrument['max_qty'] = float(filter_info['maxQty'])
                        instrument['step_size'] = float(filter_info['stepSize'])
                
                instruments.append(instrument)
            
            return instruments
            
        except Exception as e:
            self.handle_error(e, "get_instruments")
            raise
    
    def normalize_instrument_id(self, instrument: str) -> str:
        """
        Normalize instrument identifier to Binance format.
        
        Args:
            instrument: Instrument identifier in standard format
            
        Returns:
            Instrument identifier in Binance format
        """
        # Remove slash if present (e.g., 'BTC/USDT' -> 'BTCUSDT')
        normalized = instrument.replace('/', '')
        return normalized.upper()
    
    def _process_trade_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade stream data."""
        stream_data = data['data']
        
        processed = {
            'type': 'trade',
            'symbol': stream_data['s'],
            'id': stream_data['t'],
            'price': float(stream_data['p']),
            'quantity': float(stream_data['q']),
            'buyer_order_id': stream_data['b'],
            'seller_order_id': stream_data['a'],
            'timestamp': pd.to_datetime(stream_data['T'], unit='ms'),
            'is_buyer_maker': stream_data['m'],
            'is_best_match': stream_data['M']
        }
        
        return processed
    
    def _process_kline_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process kline (candlestick) stream data."""
        stream_data = data['data']
        kline = stream_data['k']
        
        processed = {
            'type': 'kline',
            'symbol': stream_data['s'],
            'interval': kline['i'],
            'start_time': pd.to_datetime(kline['t'], unit='ms'),
            'close_time': pd.to_datetime(kline['T'], unit='ms'),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'trades': int(kline['n']),
            'is_closed': kline['x']
        }
        
        return processed
    
    def _process_ticker_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ticker stream data."""
        stream_data = data['data']
        
        processed = {
            'type': 'ticker',
            'symbol': stream_data['s'],
            'price_change': float(stream_data['p']),
            'price_change_percent': float(stream_data['P']),
            'weighted_avg_price': float(stream_data['w']),
            'prev_close_price': float(stream_data['x']),
            'last_price': float(stream_data['c']),
            'last_qty': float(stream_data['Q']),
            'bid_price': float(stream_data['b']),
            'bid_qty': float(stream_data['B']),
            'ask_price': float(stream_data['a']),
            'ask_qty': float(stream_data['A']),
            'open_price': float(stream_data['o']),
            'high_price': float(stream_data['h']),
            'low_price': float(stream_data['l']),
            'volume': float(stream_data['v']),
            'quote_volume': float(stream_data['q']),
            'open_time': pd.to_datetime(stream_data['O'], unit='ms'),
            'close_time': pd.to_datetime(stream_data['C'], unit='ms'),
            'first_trade_id': stream_data['F'],
            'last_trade_id': stream_data['L'],
            'trades': stream_data['n']
        }
        
        return processed
    
    def _process_depth_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process order book depth stream data."""
        stream_data = data['data']
        
        processed = {
            'type': 'depth',
            'symbol': stream_data['s'],
            'event_time': pd.to_datetime(stream_data['E'], unit='ms'),
            'first_update_id': stream_data['U'],
            'final_update_id': stream_data['u'],
            'bids': [[float(price), float(qty)] for price, qty in stream_data['b']],
            'asks': [[float(price), float(qty)] for price, qty in stream_data['a']]
        }
        
        return processed
    
    def _sign_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a request with API secret for authenticated endpoints.
        
        Args:
            params: Request parameters
            
        Returns:
            Parameters with signature added
        """
        if not self.api_secret:
            raise ValueError("API secret is required for authenticated endpoints")
        
        # Add timestamp if not present
        if 'timestamp' not in params:
            params['timestamp'] = int(time.time() * 1000)
        
        # Create query string
        query_string = urlencode(params)
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Add signature to parameters
        params['signature'] = signature
        
        return params