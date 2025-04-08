import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import time
import logging
import threading
import json
import hmac
import hashlib
import base64
import requests
import websocket
import ssl

from ..base_provider import BaseDataProvider
from ..utils.rate_limiter import RateLimiter, MultiRateLimiter
from ..utils.data_normalizer import DataNormalizer

class CoinbaseDataProvider(BaseDataProvider):
    """
    Data provider for Coinbase Pro exchange.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Coinbase Pro data provider.
        
        Args:
            config: Configuration dictionary with Coinbase-specific settings
        """
        super().__init__(config)
        
        # Set up API endpoints
        self.base_url = config.get('base_url', 'https://api.exchange.coinbase.com')
        self.base_wss_url = config.get('base_wss_url', 'wss://ws-feed.exchange.coinbase.com')
        
        # Set up API credentials
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.passphrase = config.get('passphrase', '')
        
        # Set up rate limiters
        self.rate_limiter = MultiRateLimiter()
        self.rate_limiter.add_limiter('public', 3, 1)  # 3 requests per second
        self.rate_limiter.add_limiter('private', 5, 1/3)  # 5 requests per 3 seconds
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Set up websocket connections
        self.ws_connections = {}
        self.ws_callbacks = {}
        self.ws_lock = threading.Lock()
        
        # Set up session for HTTP requests
        self.session = requests.Session()
    
    def connect(self) -> bool:
        """
        Establish connection to Coinbase Pro.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Test connection by fetching server time
            response = self.session.get(f"{self.base_url}/time")
            response.raise_for_status()
            self.logger.info("Connected to Coinbase Pro API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Coinbase Pro API: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Coinbase Pro.
        
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
            
            self.logger.info("Disconnected from Coinbase Pro API")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Coinbase Pro API: {str(e)}")
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
        Fetch historical market data from Coinbase Pro.
        
        Args:
            instrument: Instrument identifier (e.g., 'BTC-USD')
            start_time: Start time for historical data
            end_time: End time for historical data
            interval: Time interval for the data (e.g., '1m', '1h', '1d')
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with historical market data
        """
        # Normalize instrument identifier
        product_id = self.normalize_instrument_id(instrument)
        
        # Convert datetime objects to ISO format
        if isinstance(start_time, datetime):
            start_iso = start_time.isoformat()
        else:
            start_iso = pd.to_datetime(start_time).isoformat()
        
        if isinstance(end_time, datetime):
            end_iso = end_time.isoformat()
        else:
            end_iso = pd.to_datetime(end_time).isoformat()
        
        # Map interval to Coinbase granularity (in seconds)
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '6h': 21600,
            '1d': 86400
        }
        
        granularity = interval_map.get(interval)
        if granularity is None:
            raise ValueError(f"Unsupported interval: {interval}. Supported intervals: {list(interval_map.keys())}")
        
        # Coinbase Pro has a limit of 300 candles per request
        # We need to make multiple requests for longer time ranges
        all_candles = []
        current_start = pd.to_datetime(start_iso)
        current_end = pd.to_datetime(end_iso)
        
        # Calculate maximum time range for a single request based on granularity
        max_range_seconds = 300 * granularity
        
        while current_start < current_end:
            # Calculate end time for this chunk
            chunk_end = min(
                current_end,
                current_start + timedelta(seconds=max_range_seconds)
            )
            
            # Check rate limit
            self.rate_limiter.wait_for_token('public')
            
            # Make request
            try:
                params = {
                    'start': current_start.isoformat(),
                    'end': chunk_end.isoformat(),
                    'granularity': granularity
                }
                
                response = self.session.get(
                    f"{self.base_url}/products/{product_id}/candles",
                    params=params
                )
                response.raise_for_status()
                candles = response.json()
                
                if not candles:
                    break
                
                # Coinbase returns candles in reverse order (newest first)
                candles.reverse()
                all_candles.extend(candles)
                
                # Update start time for next chunk
                current_start = chunk_end
                
            except Exception as e:
                self.handle_error(e, "get_historical_data")
                raise
        
        # Convert to DataFrame
        if not all_candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'low', 'high', 'open', 'close', 'volume'
        ])
        
        # Convert timestamp from seconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Reorder columns to standard OHLCV format
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Clean data
        df = DataNormalizer.clean_dataframe(df)
        
        return df
    
    def get_latest_data(self, instrument: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch the latest market data for a specific instrument.
        
        Args:
            instrument: Instrument identifier (e.g., 'BTC-USD')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with latest market data
        """
        # Normalize instrument identifier
        product_id = self.normalize_instrument_id(instrument)
        
        # Check rate limit
        self.rate_limiter.wait_for_token('public')
        
        try:
            # Fetch ticker data
            response = self.session.get(f"{self.base_url}/products/{product_id}/ticker")
            response.raise_for_status()
            ticker_data = response.json()
            
            # Fetch 24h stats
            self.rate_limiter.wait_for_token('public')
            stats_response = self.session.get(f"{self.base_url}/products/{product_id}/stats")
            stats_response.raise_for_status()
            stats_data = stats_response.json()
            
            # Normalize data
            result = {
                'symbol': product_id,
                'price': float(ticker_data['price']),
                'open': float(stats_data['open']),
                'high': float(stats_data['high']),
                'low': float(stats_data['low']),
                'volume': float(stats_data['volume']),
                'bid': float(ticker_data.get('bid', 0)),
                'ask': float(ticker_data.get('ask', 0)),
                'timestamp': pd.to_datetime(ticker_data['time'])
            }
            
            return result
            
        except Exception as e:
            self.handle_error(e, "get_latest_data")
            raise
    
    def subscribe_to_stream(
        self,
        instruments: List[str],
        callback: Callable,
        stream_type: str = 'ticker',
        **kwargs
    ) -> str:
        """
        Subscribe to a real-time data stream.
        
        Args:
            instruments: List of instrument identifiers
            callback: Function to call when new data is received
            stream_type: Type of stream ('ticker', 'matches', 'level2', 'full')
            **kwargs: Additional parameters
            
        Returns:
            Stream identifier
        """
        # Normalize instrument identifiers
        product_ids = [self.normalize_instrument_id(instrument) for instrument in instruments]
        
        # Generate a unique stream ID
        stream_id = f"{stream_type}_{','.join(product_ids)}_{int(time.time())}"
        
        # Map stream_type to Coinbase Pro channels
        channel_map = {
            'ticker': 'ticker',
            'trades': 'matches',
            'orderbook': 'level2',
            'full': 'full'
        }
        
        channel = channel_map.get(stream_type, stream_type)
        
        # Prepare subscription message
        subscription = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": [channel]
        }
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Skip subscription confirmation messages
                if data.get('type') == 'subscriptions':
                    return
                
                # Process data based on stream type
                if channel == 'ticker':
                    processed_data = self._process_ticker_stream(data)
                elif channel == 'matches':
                    processed_data = self._process_matches_stream(data)
                elif channel == 'level2':
                    processed_data = self._process_level2_stream(data)
                elif channel == 'full':
                    processed_data = self._process_full_stream(data)
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
                    self._reconnect_stream(stream_id, subscription, on_message, on_error, on_close)
        
        def on_open(ws):
            self.logger.info(f"Websocket opened for stream: {stream_id}")
            # Send subscription message
            ws.send(json.dumps(subscription))
        
        # Create and start websocket
        ws = websocket.WebSocketApp(
            self.base_wss_url,
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
    
    def _reconnect_stream(self, stream_id, subscription, on_message, on_error, on_close):
        """Helper method to reconnect a dropped websocket stream."""
        try:
            def on_open(ws):
                self.logger.info(f"Reconnected stream: {stream_id}")
                # Send subscription message
                ws.send(json.dumps(subscription))
            
            ws = websocket.WebSocketApp(
                self.base_wss_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
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
        Get a list of available instruments from Coinbase Pro.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            List of instrument dictionaries with metadata
        """
        # Check rate limit
        self.rate_limiter.wait_for_token('public')
        
        try:
            # Fetch products
            response = self.session.get(f"{self.base_url}/products")
            response.raise_for_status()
            products = response.json()
            
            # Extract and normalize instrument information
            instruments = []
            for product in products:
                instrument = {
                    'symbol': product['id'],
                    'base_asset': product['base_currency'],
                    'quote_asset': product['quote_currency'],
                    'min_price': float(product.get('quote_increment', 0)),
                    'max_price': None,  # Coinbase doesn't provide this
                    'tick_size': float(product.get('quote_increment', 0)),
                    'min_qty': float(product.get('base_min_size', 0)),
                    'max_qty': float(product.get('base_max_size', float('inf'))),
                    'step_size': float(product.get('base_increment', 0)),
                    'status': product['status']
                }
                
                instruments.append(instrument)
            
            return instruments
            
        except Exception as e:
            self.handle_error(e, "get_instruments")
            raise
    
    def normalize_instrument_id(self, instrument: str) -> str:
        """
        Normalize instrument identifier to Coinbase Pro format.
        
        Args:
            instrument: Instrument identifier in standard format
            
        Returns:
            Instrument identifier in Coinbase Pro format
        """
        # Replace slash with dash if present (e.g., 'BTC/USD' -> 'BTC-USD')
        normalized = instrument.replace('/', '-')
        return normalized.upper()
    
    def _process_ticker_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ticker stream data."""
        processed = {
            'type': 'ticker',
            'symbol': data['product_id'],
            'price': float(data['price']),
            'open_24h': float(data.get('open_24h', 0)),
            'volume_24h': float(data.get('volume_24h', 0)),
            'low_24h': float(data.get('low_24h', 0)),
            'high_24h': float(data.get('high_24h', 0)),
            'best_bid': float(data.get('best_bid', 0)),
            'best_ask': float(data.get('best_ask', 0)),
            'side': data.get('side', ''),
            'trade_id': data.get('trade_id', 0),
            'timestamp': pd.to_datetime(data['time'])
        }
        
        return processed
    
    def _process_matches_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process matches (trades) stream data."""
        processed = {
            'type': 'trade',
            'symbol': data['product_id'],
            'id': data['trade_id'],
            'price': float(data['price']),
            'size': float(data['size']),
            'side': data['side'],  # 'buy' or 'sell'
            'timestamp': pd.to_datetime(data['time'])
        }
        
        return processed
    
    def _process_level2_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process level2 (order book) stream data."""
        if data['type'] == 'snapshot':
            # Initial snapshot of the order book
            processed = {
                'type': 'orderbook_snapshot',
                'symbol': data['product_id'],
                'bids': [[float(price), float(size)] for price, size in data['bids']],
                'asks': [[float(price), float(size)] for price, size in data['asks']],
                'timestamp': pd.to_datetime('now', utc=True)
            }
        else:  # 'l2update'
            # Update to the order book
            processed = {
                'type': 'orderbook_update',
                'symbol': data['product_id'],
                'changes': [
                    [change[0], float(change[1]), float(change[2])]
                    for change in data['changes']
                ],
                'timestamp': pd.to_datetime(data['time'])
            }
        
        return processed
    
    def _process_full_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process full channel data (orders, matches, etc.)."""
        # The full channel includes various message types
        # We'll just pass through the data with minimal processing
        processed = {
            'type': data['type'],
            'symbol': data.get('product_id', ''),
            'timestamp': pd.to_datetime(data.get('time', 'now'), utc=True),
            'data': data
        }
        
        return processed
    
    def _generate_auth_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """
        Generate authentication headers for private API endpoints.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            request_path: Request path
            body: Request body (for POST requests)
            
        Returns:
            Dictionary with authentication headers
        """
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise ValueError("API credentials are required for authenticated endpoints")
        
        timestamp = str(time.time())
        message = timestamp + method + request_path + body
        
        # Base64 decode the secret
        secret = base64.b64decode(self.api_secret)
        
        # Create signature using HMAC-SHA256
        signature = hmac.new(
            secret,
            message.encode('utf-8'),
            hashlib.sha256
        )
        signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')
        
        # Create headers
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        return headers