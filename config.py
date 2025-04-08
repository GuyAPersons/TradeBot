import MetaTrader5 as mt5

# MT5 Account Configuration
MT5_ACCOUNT = {
    "username": "12345678",  # Replace with your account number
    "password": "your_password",  # Replace with your password
    "server": "your_server",  # Replace with your server name
    "timeout": 60000,  # Connection timeout in milliseconds
    "portable": False  # Set to True if using portable version
}

# Crypto Wallet Configuration
CRYPTO_SETTINGS = {
    "wallet_type": "metamask",  # Options: "metamask", "trustwallet", "ledger", "trezor"
    "network": "ethereum",  # Default network
    "gas_limit": 250000,
    "slippage_tolerance": 0.5,  # 0.5%
    "max_gas_price": 100,  # Maximum gas price in Gwei
    "use_flashbots": False,  # Use Flashbots to prevent frontrunning
    "rpc_endpoints": {
        "ethereum": "https://mainnet.infura.io/v3/YOUR_INFURA_KEY",
        "bsc": "https://bsc-dataseed.binance.org/",
        "polygon": "https://polygon-rpc.com",
        "arbitrum": "https://arb1.arbitrum.io/rpc",
        "optimism": "https://mainnet.optimism.io",
        "avalanche": "https://api.avax.network/ext/bc/C/rpc",
        "fantom": "https://rpc.ftm.tools/"
    }
}

# Trading Parameters
TRADING_PARAMS = {
    # Risk Management
    "max_daily_drawdown": 3.0,  # Maximum daily drawdown percentage
    "max_total_drawdown": 8.0,  # Maximum total drawdown percentage
    "risk_per_trade": 1.0,  # Risk percentage per trade
    "max_open_positions": 5,  # Maximum number of open positions
    "max_leverage": 10,  # Maximum leverage to use
    "use_stop_loss": True,  # Always use stop loss
    "use_take_profit": True,  # Always use take profit
    "dynamic_position_sizing": True,  # Adjust position size based on volatility
    
    # Strategy Settings
    "use_hedging": True,  # Enable hedging strategies
    "use_scalping": True,  # Enable scalping strategies
    "use_momentum": True,  # Enable momentum trading strategies
    "use_mean_reversion": True,  # Enable mean reversion trading strategies
    "use_volatility": True,  # Enable volatility trading strategies
    "use_trend_following": True,  # Enable trend following trading strategies
    
    # Compliance Settings
    "prop_firm_mode": True,  # Enable prop firm compliance mode
    "prop_firm_rules": {
        "max_daily_loss": 5.0,  # Maximum daily loss percentage
        "max_total_loss": 10.0,  # Maximum total loss percentage
        "no_overnight_positions": True,  # No positions held overnight
        "no_weekend_positions": True,  # No positions held over weekend
        "max_position_size": 2.0  # Maximum position size as percentage of account
    },
    
    # Execution Settings
    "use_market_orders": False,  # Use market orders (True) or limit orders (False)
    "retry_failed_orders": 3,  # Number of times to retry failed orders
    "order_timeout": 30  # Seconds to wait for order confirmation
}

# Timeframes for analysis
TIMEFRAMES = [
    mt5.TIMEFRAME_M1,
    mt5.TIMEFRAME_M5,
    mt5.TIMEFRAME_M15,
    mt5.TIMEFRAME_M30,
    mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H4,
    mt5.TIMEFRAME_D1,
    mt5.TIMEFRAME_W1
]

# Trading Instruments
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "GBPJPY", 
    "USDCHF", "EURGBP", "EURJPY", "NZDUSD", "AUDJPY", "EURCHF"
]

CRYPTO_PAIRS = [
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "ADAUSD", "DOTUSD", 
    "SOLUSD", "AVAXUSD", "MATICUSD", "LINKUSD", "DOGEUSD", "UNIUSD"
]

INDICES = [
    "US30", "US500", "USTEC", "UK100", "GER40", "FRA40", 
    "AUS200", "JPN225", "HK50", "ESP35"
]

COMMODITIES = [
    "GOLD", "SILVER", "OIL", "NATGAS", "COPPER", "PLATINUM", 
    "PALLADIUM", "WHEAT", "CORN", "COTTON"
]

TRADING_INSTRUMENTS = FOREX_PAIRS + CRYPTO_PAIRS + INDICES + COMMODITIES

DEFI_TOKENS = {
    "ethereum": [
        {"symbol": "ETH", "address": "native"},
        {"symbol": "USDT", "address": "0xdac17f958d2ee523a2206206994597c13d831ec7"},
        {"symbol": "USDC", "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"},
        {"symbol": "WBTC", "address": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"},
        {"symbol": "DAI", "address": "0x6b175474e89094c44da98b954eedeac495271d0f"},
        {"symbol": "LINK", "address": "0x514910771af9ca656af840dff83e8264ecf986ca"},
        {"symbol": "UNI", "address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"},
        {"symbol": "AAVE", "address": "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9"},
        {"symbol": "YFI", "address": "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e"},
        {"symbol": "COMP", "address": "0xc00e94cb662c3520282e6f5717214004a7f26888"},
        {"symbol": "SUSHI", "address": "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2"},
        {"symbol": "MATIC", "address": "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0"},
        {"symbol": "DOT", "address": "0x7083609fce4d1d8dc0c979aab8c869ea2c873402"},
        {"symbol": "LTC", "address": "0x4338665cbb7b2485a8855a139b75d5e34ab0db94"},
        {"symbol": "XRP", "address": "0x1d2f0da169ceb9fc7b3144628db156f3f6c60dbe"},
        {"symbol": "BCH", "address": "0x8ff795a6f4d97e7887c79bea79aba5cc7fba9248"},
        {"symbol": "ETC", "address": "0x85d1d9215bdeef092420a3552cff215c93660b98"},
        {"symbol": "XLM", "address": "0x90c97f71e18723b0cf0dfa30ee176ab653e89f40"},
        {"symbol": "XMR", "address": "0x516ffd7d1e0ca4d1ee170d12c2a21c537343834a"},
        {"symbol": "ZEC", "address": "0x1613beb3b2c4f22eef572deb9941a42f8c766748"},
        {"symbol": "DASH", "address": "0x7c025200d36ceb5a89e21eb2ce2bdc4727059d73"},
        {"symbol": "LUNA", "address": "0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3"},
        {"symbol": "ATOM", "address": "0x0970b9b0f758875a45168711cf037f0260ea64e2"},
        {"symbol": "SOL", "address": "0x639cb7b24841318f2856022f83a4668650bcafd9"},
        {"symbol": "AVAX", "address": "0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7"},
        {"symbol": "ALGO", "address": "0x6ce8dade5900945b82c5c303337718bb6d1820"},
        {"symbol": "MANA", "address": "0x0f5d2fb29fb7d3cfee444a200298f468908cc942"},
        {"symbol": "DOGE", "address": "0xbA2aE424d960c26247Dd6c32edC70B295c744C43"},
        {"symbol": "SHIB", "address": "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE"},
        {"symbol": "CRO", "address": "0xA0b73E1Ff0B80914AB6fe0444E65848C4C34450b"},
        {"symbol": "FTM", "address": "0x4E15361FD6b4BB609Fa63C81A2be19d873717870"},
        {"symbol": "SAND", "address": "0x3845badAde8e6dFF049820680d1F14bD3903a5d0"},
        {"symbol": "AXS", "address": "0xBB0E17EF65F82Ab018d8EDd776e8DD940327B28b"},
        {"symbol": "GRT", "address": "0xc944E90C64B2c07662A292be6244BDf05Cda44a7"},
        {"symbol": "SNX", "address": "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F"},
        {"symbol": "1INCH", "address": "0x111111111117dC0aa78b770fA6A738034120C302"}
    ],
    "bsc": [
        {"symbol": "BNB", "address": "native"},
        {"symbol": "BUSD", "address": "0xe9e7cea3dedca5984780bafc599bd69add087d56"},
        {"symbol": "CAKE", "address": "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82"},
        {"symbol": "USDT", "address": "0x55d398326f99059ff775485246999027b3197955"},
        {"symbol": "BTCB", "address": "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c"},
        {"symbol": "ETH", "address": "0x2170ed0880ac9a755fd29b2688956bd959f933f8"},
        {"symbol": "DOT", "address": "0x7083609fce4d1d8dc0c979aab8c869ea2c873402"},
        {"symbol": "ADA", "address": "0x3ee2200efb3400fabb9aacf31297cbdd1d435d47"},
        {"symbol": "XRP", "address": "0x1d2f0da169ceb9fc7b3144628db156f3f6c60dbe"},
        {"symbol": "LTC", "address": "0x4338665cbb7b2485a8855a139b75d5e34ab0db94"},
        {"symbol": "LINK", "address": "0xf8a0bf9cf54bb92f17374d9e9a321e6a111a51bd"},
        {"symbol": "DOGE", "address": "0xba2ae424d960c26247dd6c32edc70b295c744c43"},
        {"symbol": "MATIC", "address": "0xcc42724c6683b7e57334c4e856f4c9965ed682bd"},
        {"symbol": "FIL", "address": "0x0d8ce2a99bb6e3b7db580ed848240e4a0f9ae153"},
        {"symbol": "TRX", "address": "0x85eac5ac2f758618dfa09bdbe0cf174e7d574d5b"},
        {"symbol": "SHIB", "address": "0x2859e4544c4bb03966803b044a93563bd2d0dd4d"}
    ],
    "polygon": [
        {"symbol": "MATIC", "address": "native"},
        {"symbol": "USDT", "address": "0xc2132d05d31c914a87c6611c10748aeb04b58e8f"},
        {"symbol": "USDC", "address": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"},
        {"symbol": "WBTC", "address": "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6"},
        {"symbol": "DAI", "address": "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063"},
        {"symbol": "LINK", "address": "0xf97f4df75117a78c1a5a0dbb814af92458539fb4"},
        {"symbol": "UNI", "address": "0xb33eaad8d922b1083446dc23f610c2567fb5180f"},
        {"symbol": "AAVE", "address": "0xd6df932a45c0f255f85145f286ea0b292b21c90b"},
        {"symbol": "WETH", "address": "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619"},
        {"symbol": "QUICK", "address": "0x831753dd7087cac61ab5644b308642cc1c33dc13"},
        {"symbol": "SUSHI", "address": "0x0b3f868e0be5597d5db7feb59e1cadbb0fdda50a"},
        {"symbol": "CRV", "address": "0x172370d5cd63279efa6d502dab29171933a610af"},
        {"symbol": "BAL", "address": "0x9a71012b13ca4d3d0cdc72a177df3ef03b0e76a3"},
        {"symbol": "GHST", "address": "0x385eeac5cb85a38a9a07a70c73e0a3271cfb54a7"},
        {"symbol": "FRAX", "address": "0x104592a158490a9228070e0a8e5343b499e125d0"},
        {"symbol": "FXS", "address": "0x3e121107f6f22da4911079845a470757af4e1a1b"}
    ],
    "avalanche": [
        {"symbol": "AVAX", "address": "native"},
        {"symbol": "USDT", "address": "0x9702230a8ea53601f5cd2dc00fdbc13d4df4a8c7"},
        {"symbol": "USDC", "address": "0xb97ef9ef8734c71904d8002f8b6bc66dd9c48a6e"},
        {"symbol": "WBTC", "address": "0x50b7545627a5162f82a992c33b87adc75187b218"},
        {"symbol": "WETH", "address": "0x49d5c2bdffac6ce2bfdb6640f4f80f226bc10bab"},
        {"symbol": "DAI", "address": "0xd586e7f844cea2f87f50152665bcbc2c279d8d70"},
        {"symbol": "LINK", "address": "0x5947bb275c521040051d82396192181b413227a3"},
        {"symbol": "JOE", "address": "0x6e84a6216ea6dacc71ee8e6b0a5b7322eebc0fdd"},
        {"symbol": "QI", "address": "0x8729438eb15e2c8b576fcc6aecda6a148776c0f5"},
        {"symbol": "TIME", "address": "0xb54f16fb19478766a268f172c9480f8da1a7c9c3"},
        {"symbol": "MIM", "address": "0x130966628846bfd36ff31a822705796e8cb8c18d"}
    ],
    "fantom": [
        {"symbol": "FTM", "address": "native"},
        {"symbol": "USDC", "address": "0x04068da6c83afcfa0e13ba15a6696662335d5b75"},
        {"symbol": "WBTC", "address": "0x321162cd933e2be498cd2267a90534a804051b11"},
        {"symbol": "WETH", "address": "0x74b23882a30290451a17c44f4f05243b6b58c76d"},
        {"symbol": "DAI", "address": "0x8d11ec38a3eb5e956b052f67da8bdc9bef8abf3e"},
        {"symbol": "SPIRIT", "address": "0x5cc61a78f164885776aa610fb0fe1257df78e59b"},
        {"symbol": "BOO", "address": "0x841fad6eae12c286d1fd18d1d525dffa75c7effe"},
        {"symbol": "TOMB", "address": "0x6c021ae822bea943b2e66552bde1d2696a53fbb7"}
    ],
    "arbitrum": [
        {"symbol": "ETH", "address": "native"},
        {"symbol": "USDC", "address": "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8"},
        {"symbol": "USDT", "address": "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9"},
        {"symbol": "WBTC", "address": "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f"},
        {"symbol": "DAI", "address": "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1"},
        {"symbol": "LINK", "address": "0xf97f4df75117a78c1a5a0dbb814af92458539fb4"},
        {"symbol": "GMX", "address": "0xfc5a1a6eb076a2c7ad06ed22c90d7e710e35ad0a"},
        {"symbol": "DPX", "address": "0x6c2c06790b3e3e3c38e12ee22f8183b37a13ee55"}
    ],
    "optimism": [
        {"symbol": "ETH", "address": "native"},
        {"symbol": "USDC", "address": "0x7f5c764cbc14f9669b88837ca1490cca17c31607"},
        {"symbol": "USDT", "address": "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58"},
        {"symbol": "DAI", "address": "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1"},
        {"symbol": "WBTC", "address": "0x68f180fcce6836688e9084f035309e29bf0a2095"},
        {"symbol": "OP", "address": "0x4200000000000000000000000000000000000042"},
        {"symbol": "SNX", "address": "0x8700daec35af8ff88c16bdf0418a1a7ff238754b"},
        {"symbol": "SUSD", "address": "0x8c6f28f2f1a3c87f0f938b96d27520d9751ec8d9"},
        {"symbol": "PERP", "address": "0x9e1028f5f1d5ede59748ffcee5532509976840e0"},
        {"symbol": "LYRA", "address": "0x50c5725949a6f0c72e6c4a641f24049a917db0cb"}
    ],
    "base": [
        {"symbol": "ETH", "address": "native"},
        {"symbol": "USDC", "address": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"},
        {"symbol": "DAI", "address": "0x50c5725949a6f0c72e6c4a641f24049a917db0cb"},
        {"symbol": "WETH", "address": "0x4200000000000000000000000000000000000006"},
        {"symbol": "USDbC", "address": "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca"},
        {"symbol": "cbETH", "address": "0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22"}
    ],
    "solana": [
        {"symbol": "SOL", "address": "native"},
        {"symbol": "USDC", "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
        {"symbol": "USDT", "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"},
        {"symbol": "BTC", "address": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"},
        {"symbol": "ETH", "address": "2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Pxk"},
        {"symbol": "RAY", "address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"},
        {"symbol": "SRM", "address": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt"},
        {"symbol": "MNGO", "address": "MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac"},
        {"symbol": "ORCA", "address": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE"},
        {"symbol": "BONK", "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"}
    ],
    "celo": [
        {"symbol": "CELO", "address": "native"},
        {"symbol": "cUSD", "address": "0x765de816845861e75a25fca122bb6898b8b1282a"},
        {"symbol": "cEUR", "address": "0xd8763cba276a3738e6de85b4b3bf5fded6d6ca73"},
        {"symbol": "cREAL", "address": "0xe8537a3d056da446677b9e9d6c5db704eaab4787"},
        {"symbol": "WETH", "address": "0x2def4285787d58a2f811af24755a8150622f4361"},
        {"symbol": "WBTC", "address": "0xbaab46e28388d2779e6e31fd00cf0e5ad95e327b"}
    ],
    "gnosis": [
        {"symbol": "xDAI", "address": "native"},
        {"symbol": "USDC", "address": "0xddafbb505ad214d7b80b1f830fccc89b60fb7a83"},
        {"symbol": "USDT", "address": "0x4ecaba5870353805a9f068101a40e0f32ed605c6"},
        {"symbol": "WETH", "address": "0x6a023ccd1ff6f2045c3309768ead9e68f978f6e1"},
        {"symbol": "GNO", "address": "0x9c58bacc331c9aa871afd802db6379a98e80cedb"},
        {"symbol": "WBTC", "address": "0x8e5bbbb09ed1ebde8674cda39a0c169401db4252"}
    ],
    "zksync": [
        {"symbol": "ETH", "address": "native"},
        {"symbol": "USDC", "address": "0x3355df6d4c9c3035724fd0e3914de96a5a83aaf4"},
        {"symbol": "USDT", "address": "0x493257fd37edb34451f62edf8d2a0c418852ba4c"},
        {"symbol": "WBTC", "address": "0xbbeb516fb02a01611cbbe0453fe3c580d7281011"},
        {"symbol": "DAI", "address": "0x4b9eb6c0b6ea15176bbf62841c6b2a8a398cb656"},
        {"symbol": "LINK", "address": "0x40609141db628beee3bfab8034fc2d8278d0cc78"}
    ],
    "linea": [
        {"symbol": "ETH", "address": "native"},
        {"symbol": "USDC", "address": "0x176211869ca2b568f2a7d4ee941e073a821ee1ff"},
        {"symbol": "USDT", "address": "0xa219439258ca9da29e9cc4ce5596924745e12b93"},
        {"symbol": "WBTC", "address": "0x3adf83d672a546125d5f8d9f35c8f346868a0279"},
        {"symbol": "DAI", "address": "0x4af15ec2a0bd43db75dd04e62faa3b8ef36b00d5"},
        {"symbol": "LINK", "address": "0x5471ea8f739dd37e9b81be9c5c77754d8aa953e4"}
    ],
    "mantle": [
        {"symbol": "MNT", "address": "native"},
        {"symbol": "USDC", "address": "0x09bc4e0d864854c6afb6eb9a9cdf58ac190d0df9"},
        {"symbol": "USDT", "address": "0x201eba5cc46d216ce6dc03f6a759e8e766e956ae"},
        {"symbol": "WETH", "address": "0xdeaddeaddeaddeaddeaddeaddeaddeaddead0000"},
        {"symbol": "WMNT", "address": "0x78c1b0c915c4faa5fffa6cabf0219da63d7f4cb8"}
    ],
    "scroll": [
        {"symbol": "ETH", "address": "native"},
        {"symbol": "USDC", "address": "0x06efdbff2a14a7c8e15944d1f4a48f9f95f663a4"},
        {"symbol": "USDT", "address": "0xf55bec9cafdbe8730f096aa55dad6d22d44099df"},
        {"symbol": "WBTC", "address": "0x3c1bca5a656e69edcd0d4e36bebb3fcdaca60cf1"},
        {"symbol": "DAI", "address": "0xcA77eB3fEFe3725Dc33bccB54eDEFc3D9f764f97"}
    ]
}

# API Keys (store these securely in production)
API_KEYS = {
    "infura": "YOUR_INFURA_KEY",
    "etherscan": "YOUR_ETHERSCAN_KEY",
    "bscscan": "YOUR_BSCSCAN_KEY",
    "polygonscan": "YOUR_POLYGONSCAN_KEY",
    "alphavantage": "YOUR_ALPHAVANTAGE_KEY",
    "tradingview": "YOUR_TRADINGVIEW_KEY"
}

# Notification Settings
NOTIFICATION_SETTINGS = {
    "use_email": False,
    "email_address": "your_email@example.com",
    "use_telegram": False,
    "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID",
    "notify_on_trade": True,
    "notify_on_error": True,
    "notify_daily_summary": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_to_file": True,
    "log_file_path": "trading_bot.log",
    "rotate_logs": True,
    "max_log_size_mb": 10,
    "backup_count": 5
}