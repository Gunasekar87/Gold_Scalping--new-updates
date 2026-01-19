import MetaTrader5 as mt5
import logging
import json
import os
from dotenv import load_dotenv

logger = logging.getLogger("SessionManager")

class SessionManager:
    @staticmethod
    def initialize():
        # 1. Try User Config (JSON)
        config_path = "config/user_config.json"
        login = 0
        password = ""
        server = ""
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    login = int(config.get("login", 0))
                    password = config.get("password", "")
                    server = config.get("server", "")
            except Exception as e:
                logger.error(f"Error reading user_config.json: {e}")

        # 2. Fallback to Environment Variables (secrets.env)
        if not (login and password and server):
            env_path = "config/secrets.env"
            if os.path.exists(env_path):
                logger.info("Falling back to secrets.env for credentials...")
                load_dotenv(env_path)
                if not login: 
                    try:
                        login = int(os.getenv("MT5_LOGIN", 0))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid MT5_LOGIN in env: {e}")
                        login = 0
                if not password: password = os.getenv("MT5_PASSWORD", "")
                if not server: server = os.getenv("MT5_SERVER", "")

        # 3. Validate
        if not login or not password or not server:
            logger.error(f"Login Credentials Missing! Login: {login}, Server: {server}, Password: {'OK' if password else 'MISSING'}")
            logger.error("Please configure via Dashboard or config/secrets.env")
            return False

        # 4. Initialize MT5
        if not mt5.initialize():
            logger.error("MT5 Initialize failed")
            return False
        
        # 5. Login
        authorized = mt5.login(login, password=password, server=server)
        if authorized:
            logger.info(f"Connected to account #{login} on {server}")
            return True
        else:
            logger.error(f"Failed to connect to account #{login}, error code: {mt5.last_error()}")
            return False

    @staticmethod
    def get_symbols():
        config_path = "config/user_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    return config.get("symbols", ["XAUUSD"])
            except Exception as e:
                logger.warning(f"Failed to load symbols from config: {e}")
        return ["XAUUSD"]
