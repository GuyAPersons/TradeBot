import pickle
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple
import os
import gzip
import base64
import datetime
import uuid
from io import BytesIO

logger = logging.getLogger(__name__)

class BacktestSerializer:
    """
    Utility class for serializing and deserializing backtest data.
    
    This class provides methods to save and load various backtest components,
    including strategy objects, pandas DataFrames, trade history, and results.
    """
    
    @staticmethod
    def serialize_dataframe(df: pd.DataFrame, format: str = 'csv') -> Union[str, bytes]:
        """
        Serialize a pandas DataFrame to the specified format.
        
        Args:
            df: DataFrame to serialize
            format: Output format ('csv', 'json', 'pickle', 'parquet')
            
        Returns:
            Serialized DataFrame as string or bytes
        """
        if df is None or df.empty:
            logger.warning("Attempted to serialize empty DataFrame")
            return "" if format in ['csv', 'json'] else b""
        
        buffer = BytesIO()
        
        try:
            if format == 'csv':
                return df.to_csv(index=True)
            elif format == 'json':
                return df.to_json(orient='split', date_format='iso')
            elif format == 'pickle':
                pickle.dump(df, buffer)
                return buffer.getvalue()
            elif format == 'parquet':
                df.to_parquet(buffer)
                return buffer.getvalue()
            else:
                logger.error(f"Unsupported DataFrame format: {format}")
                return "" if format in ['csv', 'json'] else b""
        except Exception as e:
            logger.error(f"Error serializing DataFrame: {str(e)}")
            return "" if format in ['csv', 'json'] else b""
    
    @staticmethod
    def deserialize_dataframe(data: Union[str, bytes], format: str = 'csv') -> pd.DataFrame:
        """
        Deserialize data to a pandas DataFrame.
        
        Args:
            data: Serialized DataFrame data
            format: Input format ('csv', 'json', 'pickle', 'parquet')
            
        Returns:
            Deserialized DataFrame
        """
        if not data:
            logger.warning("Attempted to deserialize empty data")
            return pd.DataFrame()
        
        buffer = BytesIO(data if isinstance(data, bytes) else data.encode('utf-8'))
        
        try:
            if format == 'csv':
                return pd.read_csv(buffer, index_col=0)
            elif format == 'json':
                return pd.read_json(buffer, orient='split')
            elif format == 'pickle':
                return pickle.load(buffer)
            elif format == 'parquet':
                return pd.read_parquet(buffer)
            else:
                logger.error(f"Unsupported DataFrame format: {format}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error deserializing DataFrame: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], 
                      format: str = None, compress: bool = False) -> bool:
        """
        Save a DataFrame to a file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the file
            format: Output format (inferred from extension if None)
            compress: Whether to compress the output
            
        Returns:
            True if successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning(f"Attempted to save empty DataFrame to {filepath}")
            return False
        
        filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Infer format from extension if not specified
        if format is None:
            ext = filepath.suffix.lower()
            if ext == '.csv':
                format = 'csv'
            elif ext == '.json':
                format = 'json'
            elif ext == '.pkl' or ext == '.pickle':
                format = 'pickle'
            elif ext == '.parquet':
                format = 'parquet'
            else:
                format = 'csv'  # Default to CSV
        
        try:
            if compress:
                with gzip.open(str(filepath) + '.gz', 'wb') as f:
                    if format == 'csv':
                        f.write(df.to_csv(index=True).encode('utf-8'))
                    elif format == 'json':
                        f.write(df.to_json(orient='split', date_format='iso').encode('utf-8'))
                    elif format == 'pickle':
                        pickle.dump(df, f)
                    elif format == 'parquet':
                        buffer = BytesIO()
                        df.to_parquet(buffer)
                        f.write(buffer.getvalue())
            else:
                if format == 'csv':
                    df.to_csv(filepath, index=True)
                elif format == 'json':
                    df.to_json(filepath, orient='split', date_format='iso')
                elif format == 'pickle':
                    with open(filepath, 'wb') as f:
                        pickle.dump(df, f)
                elif format == 'parquet':
                    df.to_parquet(filepath)
            
            logger.info(f"DataFrame saved to {filepath}{'.gz' if compress else ''}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_dataframe(filepath: Union[str, Path], format: str = None) -> pd.DataFrame:
        """
        Load a DataFrame from a file.
        
        Args:
            filepath: Path to the file
            format: Input format (inferred from extension if None)
            
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        # Check if file exists
        if not filepath.exists():
            # Check for compressed version
            if Path(str(filepath) + '.gz').exists():
                filepath = Path(str(filepath) + '.gz')
                compressed = True
            else:
                logger.error(f"File not found: {filepath}")
                return pd.DataFrame()
        else:
            compressed = filepath.suffix.lower() == '.gz'
        
        # Infer format from extension if not specified
        if format is None:
            ext = filepath.suffix.lower()
            if compressed:
                # Remove .gz extension to get the actual format
                ext = Path(filepath.stem).suffix.lower()
            
            if ext == '.csv':
                format = 'csv'
            elif ext == '.json':
                format = 'json'
            elif ext == '.pkl' or ext == '.pickle':
                format = 'pickle'
            elif ext == '.parquet':
                format = 'parquet'
            else:
                format = 'csv'  # Default to CSV
        
        try:
            if compressed:
                with gzip.open(filepath, 'rb') as f:
                    if format == 'csv':
                        return pd.read_csv(f, index_col=0)
                    elif format == 'json':
                        return pd.read_json(f, orient='split')
                    elif format == 'pickle':
                        return pickle.load(f)
                    elif format == 'parquet':
                        buffer = BytesIO(f.read())
                        return pd.read_parquet(buffer)
            else:
                if format == 'csv':
                    return pd.read_csv(filepath, index_col=0)
                elif format == 'json':
                    return pd.read_json(filepath, orient='split')
                elif format == 'pickle':
                    with open(filepath, 'rb') as f:
                        return pickle.load(f)
                elif format == 'parquet':
                    return pd.read_parquet(filepath)
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from {filepath}: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def serialize_object(obj: Any) -> bytes:
        """
        Serialize any Python object using pickle.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized object as bytes
        """
        try:
            return pickle.dumps(obj)
        except Exception as e:
            logger.error(f"Error serializing object: {str(e)}")
            return b""
    
    @staticmethod
    def deserialize_object(data: bytes) -> Any:
        """
        Deserialize a Python object from bytes.
        
        Args:
            data: Serialized object data
            
        Returns:
            Deserialized object
        """
        if not data:
            logger.warning("Attempted to deserialize empty data")
            return None
        
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error deserializing object: {str(e)}")
            return None
    
    @staticmethod
    def save_object(obj: Any, filepath: Union[str, Path], compress: bool = False) -> bool:
        """
        Save a Python object to a file using pickle.
        
        Args:
            obj: Object to save
            filepath: Path to save the file
            compress: Whether to compress the output
            
        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(obj, f)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(obj, f)
            
            logger.info(f"Object saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving object to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_object(filepath: Union[str, Path]) -> Any:
        """
        Load a Python object from a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Loaded object
        """
        filepath = Path(filepath)
        
        # Check if file exists
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            # Check if file is compressed
            if filepath.suffix.lower() == '.gz':
                with gzip.open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            
        except Exception as e:
            logger.error(f"Error loading object from {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def serialize_to_json(obj: Any) -> str:
        """
        Serialize an object to JSON with special handling for non-serializable types.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON string
        """
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                elif isinstance(obj, pd.DataFrame):
                    return {
                        "__type__": "DataFrame",
                        "data": obj.to_json(orient='split', date_format='iso')
                    }
                elif isinstance(obj, pd.Series):
                    return {
                        "__type__": "Series",
                        "data": obj.to_json(orient='split', date_format='iso')
                    }
                elif isinstance(obj, np.ndarray):
                    return {
                        "__type__": "ndarray",
                        "data": obj.tolist()
                    }
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, uuid.UUID):
                    return str(obj)
                elif hasattr(obj, '__dict__'):
                    return {
                        "__type__": obj.__class__.__name__,
                        "module": obj.__class__.__module__,
                        "data": obj.__dict__
                    }
                return super().default(obj)
        
        try:
            return json.dumps(obj, cls=CustomJSONEncoder)
        except Exception as e:
            logger.error(f"Error serializing to JSON: {str(e)}")
            return "{}"
    
    @staticmethod
    def deserialize_from_json(json_str: str) -> Any:
        """
        Deserialize an object from JSON with special handling for custom types.
        
        Args:
            json_str: JSON string
            
        Returns:
            Deserialized object
        """
        def object_hook(obj):
            if "__type__" in obj:
                obj_type = obj["__type__"]
                
                if obj_type == "DataFrame":
                    return pd.read_json(obj["data"], orient='split')
                elif obj_type == "Series":
                    return pd.read_json(obj["data"], orient='split', typ='series')
                elif obj_type == "ndarray":
                    return np.array(obj["data"])
                elif "module" in obj and "data" in obj:
                    try:
                        module_name = obj["module"]
                        class_name = obj_type
                        
                        # Import the module and get the class
                        import importlib
                        module = importlib.import_module(module_name)
                        cls = getattr(module, class_name)
                        
                        # Create an instance and set attributes
                        instance = cls.__new__(cls)
                        for key, value in obj["data"].items():
                            setattr(instance, key, value)
                        
                        return instance
                    except Exception as e:
                        logger.error(f"Error deserializing custom object: {str(e)}")
                        return obj
            
            return obj
        
        try:
            return json.loads(json_str, object_hook=object_hook)
        except Exception as e:
            logger.error(f"Error deserializing from JSON: {str(e)}")
            return None
    
    @staticmethod
    def save_backtest_results(results: Dict[str, Any], 
                             output_dir: Union[str, Path],
                             backtest_id: str = None) -> str:
        """
        Save backtest results to files.
        
        Args:
            results: Dictionary of backtest results
            output_dir: Directory to save results
            backtest_id: Unique identifier for the backtest (generated if None)
            
        Returns:
            Backtest ID
        """
        output_dir = Path(output_dir)
        
        # Generate backtest ID if not provided
        if backtest_id is None:
            backtest_id = f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create backtest directory
        backtest_dir = output_dir / backtest_id
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save metadata
            metadata = {
                'backtest_id': backtest_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'description': results.get('description', ''),
                'strategy_name': results.get('strategy_name', ''),
                'parameters': results.get('parameters', {}),
                'symbols': results.get('symbols', []),
                'timeframe': results.get('timeframe', ''),
                'start_date': results.get('start_date', ''),
                'end_date': results.get('end_date', ''),
                'metrics': results.get('metrics', {})
            }
            
            with open(backtest_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Save equity curve
            if 'equity_curve' in results and not results['equity_curve'].empty:
                BacktestSerializer.save_dataframe(
                    results['equity_curve'], 
                    backtest_dir / 'equity_curve.csv'
                )
            
            # Save trades
            if 'trades' in results and not results['trades'].empty:
                BacktestSerializer.save_dataframe(
                    results['trades'], 
                    backtest_dir / 'trades.csv'
                )
            
            # Save positions
            if 'positions' in results and not results['positions'].empty:
                BacktestSerializer.save_dataframe(
                    results['positions'], 
                    backtest_dir / 'positions.csv'
                )
            
            # Save drawdowns
            if 'drawdowns' in results and not results['drawdowns'].empty:
                BacktestSerializer.save_dataframe(
                    results['drawdowns'], 
                    backtest_dir / 'drawdowns.csv'
                )
            
            # Save performance metrics
            if 'metrics' in results:
                with open(backtest_dir / 'metrics.json', 'w') as f:
                    json.dump(results['metrics'], f, indent=4)
            
            # Save strategy state if available
            if 'strategy_state' in results:
                BacktestSerializer.save_object(
                    results['strategy_state'],
                    backtest_dir / 'strategy_state.pkl'
                )
            
            logger.info(f"Backtest results saved to {backtest_dir}")
            return backtest_id
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
            return backtest_id
    
    @staticmethod
    def load_backtest_results(backtest_id: str, 
                             base_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Load backtest results from files.
        
        Args:
            backtest_id: Backtest identifier
            base_dir: Base directory containing backtest results
            
        Returns:
            Dictionary of backtest results
        """
        base_dir = Path(base_dir)
        backtest_dir = base_dir / backtest_id
        
        if not backtest_dir.exists():
            logger.error(f"Backtest directory not found: {backtest_dir}")
            return {}
        
        results = {}
        
        try:
            # Load metadata
            metadata_path = backtest_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    results.update(json.load(f))
            
            # Load equity curve
            equity_path = backtest_dir / 'equity_curve.csv'
            if equity_path.exists():
                results['equity_curve'] = BacktestSerializer.load_dataframe(equity_path)
            
            # Load trades
            trades_path = backtest_dir / 'trades.csv'
            if trades_path.exists():
                results['trades'] = BacktestSerializer.load_dataframe(trades_path)
            
            # Load positions
            positions_path = backtest_dir / 'positions.csv'
            if positions_path.exists():
                results['positions'] = BacktestSerializer.load_dataframe(positions_path)
            
            # Load drawdowns
            drawdowns_path = backtest_dir / 'drawdowns.csv'
            if drawdowns_path.exists():
                results['drawdowns'] = BacktestSerializer.load_dataframe(drawdowns_path)
            
            # Load metrics
            metrics_path = backtest_dir / 'metrics.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    results['metrics'] = json.load(f)
            
            # Load strategy state
            state_path = backtest_dir / 'strategy_state.pkl'
            if state_path.exists():
                results['strategy_state'] = BacktestSerializer.load_object(state_path)
            
            logger.info(f"Backtest results loaded from {backtest_dir}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading backtest results: {str(e)}")
            return results
    
    @staticmethod
    def list_backtests(base_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        List all available backtest results in the base directory.
        
        Args:
            base_dir: Base directory containing backtest results
            
        Returns:
            List of backtest metadata dictionaries
        """
        base_dir = Path(base_dir)
        
        if not base_dir.exists():
            logger.error(f"Base directory not found: {base_dir}")
            return []
        
        backtests = []
        
        for backtest_dir in base_dir.iterdir():
            if backtest_dir.is_dir():
                metadata_path = backtest_dir / 'metadata.json'
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            backtests.append(metadata)
                    except Exception as e:
                        logger.error(f"Error loading metadata from {metadata_path}: {str(e)}")
        
        # Sort by timestamp (newest first)
        backtests.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return backtests
    
    @staticmethod
    def export_backtest_report(backtest_id: str, 
                              base_dir: Union[str, Path],
                              output_file: Union[str, Path] = None,
                              format: str = 'html') -> bool:
        """
        Export a backtest report in the specified format.
        
        Args:
            backtest_id: Backtest identifier
            base_dir: Base directory containing backtest results
            output_file: Output file path (generated if None)
            format: Output format ('html', 'pdf', 'json')
            
        Returns:
            True if successful, False otherwise
        """
        base_dir = Path(base_dir)
        backtest_dir = base_dir / backtest_id
        
        if not backtest_dir.exists():
            logger.error(f"Backtest directory not found: {backtest_dir}")
            return False
        
        # Load backtest results
        results = BacktestSerializer.load_backtest_results(backtest_id, base_dir)
        
        if not results:
            logger.error(f"Failed to load backtest results for {backtest_id}")
            return False
        
        # Generate output file path if not provided
        if output_file is None:
            output_file = backtest_dir / f"report.{format}"
        else:
            output_file = Path(output_file)
        
        try:
            if format == 'json':
                # Export as JSON
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=4, default=str)
                
            elif format == 'html':
                # Export as HTML
                try:
                    import jinja2
                    
                    # Load template
                    template_dir = Path(__file__).parent / 'templates'
                    if not template_dir.exists():
                        template_dir = Path(__file__).parent.parent / 'templates'
                    
                    env = jinja2.Environment(
                        loader=jinja2.FileSystemLoader(template_dir)
                    )
                    
                    template = env.get_template('backtest_report.html')
                    
                    # Render template
                    html = template.render(
                        backtest_id=backtest_id,
                        timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        results=results
                    )
                    
                    # Write to file
                    with open(output_file, 'w') as f:
                        f.write(html)
                    
                except ImportError:
                    logger.error("Jinja2 is required for HTML export")
                    return False
                
            elif format == 'pdf':
                # Export as PDF
                try:
                    import pdfkit
                    
                    # First export as HTML
                    html_file = output_file.with_suffix('.html')
                    if not BacktestSerializer.export_backtest_report(
                        backtest_id, base_dir, html_file, 'html'
                    ):
                        return False
                    
                    # Convert HTML to PDF
                    pdfkit.from_file(str(html_file), str(output_file))
                    
                except ImportError:
                    logger.error("pdfkit is required for PDF export")
                    return False
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Backtest report exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting backtest report: {str(e)}")
            return False