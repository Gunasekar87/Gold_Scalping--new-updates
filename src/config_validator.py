"""
Configuration Validation Module for AETHER Trading System.

Ensures all configuration files are valid before system startup, preventing
runtime errors and providing clear feedback on configuration issues.

Author: AETHER Development Team
License: MIT
Version: 2.0.0
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
import re

from src.constants import (
    PRIME_CRYPTO, PRIME_FOREX, PRIME_COMMODITIES, PRIME_INDICES,
    RiskLimits, AIConfig, CouncilThresholds
)
from src.exceptions import ConfigurationError


logger = logging.getLogger("ConfigValidator")


class ConfigValidator:
    """
    Validates all configuration files for AETHER system.
    
    Performs comprehensive validation of:
    - settings.yaml (trading parameters, risk limits)
    - secrets.env (credentials, API keys)
    - model_config.json (AI model settings)
    
    Attributes:
        strict_mode: If True, raises exceptions on warnings. If False, logs warnings only.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize configuration validator.
        
        Args:
            strict_mode: Whether to raise exceptions on non-critical issues
        """
        self.strict_mode = strict_mode
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_all(self, config_dir: str = "config") -> Tuple[bool, List[str], List[str]]:
        """
        Validate all configuration files.
        
        Args:
            config_dir: Path to configuration directory
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        config_path = Path(config_dir)
        
        # Validate settings.yaml
        settings_file = config_path / "settings.yaml"
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = yaml.safe_load(f)
                self._validate_settings(settings)
        else:
            self.errors.append(f"Critical: settings.yaml not found at {settings_file}")
        
        # Validate secrets.env
        secrets_file = config_path / "secrets.env"
        if secrets_file.exists():
            self._validate_secrets(secrets_file)
        else:
            self.errors.append(f"Critical: secrets.env not found at {secrets_file}")
        
        # Validate model_config.json (optional)
        model_config_file = config_path / "model_config.json"
        if model_config_file.exists():
            with open(model_config_file, 'r') as f:
                model_config = json.load(f)
                self._validate_model_config(model_config)
        else:
            self.warnings.append("model_config.json not found (using defaults)")
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_settings(self, settings: Dict[str, Any]) -> None:
        """Validate settings.yaml configuration."""
        
        # Validate trading section
        if 'trading' not in settings:
            self.errors.append("Missing 'trading' section in settings.yaml")
            return
        
        trading = settings['trading']
        
        # Symbol validation
        symbol = trading.get('symbol', '').upper()
        if not symbol:
            self.errors.append("trading.symbol is required")
        else:
            # Check if symbol is in prime universe
            all_prime = PRIME_CRYPTO + PRIME_FOREX + PRIME_COMMODITIES + PRIME_INDICES
            is_prime = any(s in symbol for s in all_prime)
            if not is_prime:
                self.errors.append(
                    f"Symbol '{symbol}' not in prime universe. "
                    f"Allowed: {all_prime}"
                )
        
        # Timeframe validation
        timeframe = trading.get('timeframe')
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        if timeframe not in valid_timeframes:
            self.errors.append(
                f"Invalid timeframe '{timeframe}'. Valid options: {valid_timeframes}"
            )
        
        # Magic number validation
        magic = trading.get('magic_number')
        if magic is None or not isinstance(magic, int):
            self.warnings.append("magic_number not set or invalid, using default")
        elif magic < 100000 or magic > 999999:
            self.warnings.append(f"magic_number {magic} outside recommended range 100000-999999")
        
        # Validate risk section
        if 'risk' not in settings:
            self.errors.append("Missing 'risk' section in settings.yaml")
            return
        
        risk = settings['risk']
        
        # Validate risk percentage
        risk_pct = risk.get('global_risk_percent')
        if risk_pct is None:
            self.errors.append("risk.global_risk_percent is required")
        elif not (0 < risk_pct <= RiskLimits.MAX_EQUITY_RISK_PCT * 100):
            self.errors.append(
                f"global_risk_percent {risk_pct}% exceeds maximum "
                f"{RiskLimits.MAX_EQUITY_RISK_PCT * 100}%"
            )
        
        # Validate zone recovery settings
        zone_recovery = risk.get('zone_recovery', {})
        zone_pips = zone_recovery.get('zone_pips')
        tp_pips = zone_recovery.get('tp_pips')
        
        if zone_pips is not None and (zone_pips < 5 or zone_pips > 500):
            self.warnings.append(f"zone_pips {zone_pips} outside typical range 5-500")
        
        if tp_pips is not None and (tp_pips < 5 or tp_pips > 500):
            self.warnings.append(f"tp_pips {tp_pips} outside typical range 5-500")
        
        # Validate AI parameters
        if 'ai_parameters' in settings:
            ai_params = settings['ai_parameters']
            
            nexus_confidence = ai_params.get('nexus_confidence_threshold')
            if nexus_confidence is not None:
                if not (0 < nexus_confidence < 1):
                    self.errors.append(
                        f"nexus_confidence_threshold {nexus_confidence} must be between 0 and 1"
                    )
            
            atr_zone_mult = ai_params.get('atr_zone_multiplier')
            if atr_zone_mult is not None and (atr_zone_mult < 0.1 or atr_zone_mult > 5.0):
                self.warnings.append(
                    f"atr_zone_multiplier {atr_zone_mult} outside typical range 0.1-5.0"
                )
            
            atr_tp_mult = ai_params.get('atr_tp_multiplier')
            if atr_tp_mult is not None and (atr_tp_mult < 0.1 or atr_tp_mult > 5.0):
                self.warnings.append(
                    f"atr_tp_multiplier {atr_tp_mult} outside typical range 0.1-5.0"
                )
    
    def _validate_secrets(self, secrets_file: Path) -> None:
        """Validate secrets.env file."""
        
        required_keys = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
        optional_keys = ['SUPABASE_URL', 'SUPABASE_KEY', 'API_SECRET_KEY']
        
        # Read .env file manually (don't use dotenv here to avoid loading)
        env_vars = {}
        with open(secrets_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        
        # Check required keys
        for key in required_keys:
            if key not in env_vars or not env_vars[key]:
                self.errors.append(f"Missing or empty required credential: {key}")
        
        # Validate MT5 login format
        if 'MT5_LOGIN' in env_vars:
            try:
                login = int(env_vars['MT5_LOGIN'])
                if login < 1000:
                    self.warnings.append("MT5_LOGIN seems unusually short")
            except ValueError:
                self.errors.append("MT5_LOGIN must be a valid integer")
        
        # Check if password is placeholder
        if 'MT5_PASSWORD' in env_vars:
            password = env_vars['MT5_PASSWORD']
            if password in ['your_password', 'change_me', '']:
                self.errors.append("MT5_PASSWORD appears to be a placeholder")
        
        # Validate Supabase URL format
        if 'SUPABASE_URL' in env_vars:
            url = env_vars['SUPABASE_URL']
            if url and not url.startswith('https://'):
                self.warnings.append("SUPABASE_URL should start with https://")
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """Validate model_config.json file."""
        
        # Validate Nexus Brain config
        if 'nexus_transformer' in model_config:
            nexus = model_config['nexus_transformer']
            
            seq_len = nexus.get('sequence_length')
            if seq_len and seq_len != AIConfig.NEXUS_SEQUENCE_LENGTH:
                self.warnings.append(
                    f"Nexus sequence_length {seq_len} differs from standard "
                    f"{AIConfig.NEXUS_SEQUENCE_LENGTH}"
                )
            
            num_classes = nexus.get('num_classes')
            if num_classes and num_classes != AIConfig.NEXUS_CLASSES:
                self.errors.append(
                    f"Nexus num_classes {num_classes} must be {AIConfig.NEXUS_CLASSES}"
                )
        
        # Validate PPO Guardian config
        if 'ppo_guardian' in model_config:
            ppo = model_config['ppo_guardian']
            
            obs_dim = ppo.get('observation_dim')
            if obs_dim and obs_dim != AIConfig.PPO_OBSERVATION_DIM:
                self.errors.append(
                    f"PPO observation_dim {obs_dim} must be {AIConfig.PPO_OBSERVATION_DIM}"
                )
            
            action_dim = ppo.get('action_dim')
            if action_dim and action_dim != AIConfig.PPO_ACTION_DIM:
                self.errors.append(
                    f"PPO action_dim {action_dim} must be {AIConfig.PPO_ACTION_DIM}"
                )
    
    def validate_remote_config(self, remote_config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate remote configuration from dashboard with security checks.

        Args:
            remote_config: Remote configuration dictionary

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        if not isinstance(remote_config, dict):
            errors.append("Remote config must be a dictionary")
            return False, errors, warnings

        # Validate symbol if present
        if 'symbol' in remote_config:
            symbol = remote_config['symbol']
            symbol_upper = symbol.upper()
            all_prime = PRIME_CRYPTO + PRIME_FOREX + PRIME_COMMODITIES + PRIME_INDICES
            is_prime = any(s in symbol_upper for s in all_prime)
            if not is_prime:
                errors.append(f"Remote symbol '{symbol}' not in prime universe")

        # Validate lot size if present
        if 'initial_lot' in remote_config:
            lot_size = remote_config['initial_lot']
            if not isinstance(lot_size, (int, float)) or lot_size <= 0 or lot_size > 100:
                errors.append(f"Remote initial_lot {lot_size} out of valid range (0.001-100)")

        # Validate risk level
        if 'risk_level' in remote_config:
            risk_level = remote_config['risk_level']
            if not isinstance(risk_level, (int, float)) or risk_level < 0.1 or risk_level > 5.0:
                errors.append(f"Remote risk_level {risk_level} out of range (0.1-5.0)")

        # Validate broker type
        if 'broker_type' in remote_config:
            broker_type = remote_config['broker_type']
            allowed_brokers = ["MT5", "BINANCE", "BYBIT"]
            if broker_type not in allowed_brokers:
                errors.append(f"Remote broker_type '{broker_type}' not supported")

        # Security validation - check for sensitive data exposure
        sensitive_fields = ['mt5_password', 'api_key', 'secret_key', 'password']
        for field in sensitive_fields:
            if field in remote_config and remote_config[field]:
                value = str(remote_config[field])
                # Check if it's obviously plain text (not encrypted/hashed)
                if len(value) < 20 or not any(char in value for char in ['$', '*', '#', 'eyJ']):
                    errors.append(f"SECURITY RISK: Remote {field} appears to be plain text - use encrypted values only")

        # Check for injection attempts in string fields
        injection_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
        for key, value in remote_config.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for pattern in injection_patterns:
                    if pattern in value_lower:
                        errors.append(f"SECURITY RISK: Potential injection attempt in {key}")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def sanitize_config_value(self, key: str, value: Any) -> Any:
        """
        Sanitize a configuration value to prevent injection attacks.

        Args:
            key: Configuration key
            value: Value to sanitize

        Returns:
            Sanitized value
        """
        if not isinstance(value, str):
            return value

        original_value = value

        # Remove potentially dangerous characters based on field type
        if key in ['symbol', 'broker_type', 'comment']:
            # Allow alphanumeric, dots, underscores, hyphens
            value = re.sub(r'[^a-zA-Z0-9._-]', '', value)

        elif key in ['path', 'file_path', 'model_path']:
            # Allow path characters but remove dangerous ones
            value = re.sub(r'[^a-zA-Z0-9._\-/\\:]', '', value)

        elif key in ['url', 'endpoint']:
            # Basic URL validation
            if not value.startswith(('http://', 'https://', 'ftp://')):
                value = 'https://' + value.lstrip(':/')

        if value != original_value:
            logger.warning(f"Sanitized config value for {key}: '{original_value}' -> '{value}'")

        return value

    def validate_secure_config_loading(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration for secure loading practices.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_secure, security_issues)
        """
        security_issues = []

        # Check for hardcoded credentials in config files
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        for key, value in config.items():
            key_lower = key.lower()
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                if isinstance(value, str) and len(value) > 0 and not value.startswith(('env:', 'secret:', '$')):
                    security_issues.append(f"Potentially exposed credential in config: {key}")

        # Check for dangerous file paths
        dangerous_paths = ['/etc/passwd', '/etc/shadow', 'C:\\Windows\\System32', '/bin', '/sbin']
        for key, value in config.items():
            if 'path' in key.lower() and isinstance(value, str):
                for dangerous_path in dangerous_paths:
                    if dangerous_path in value:
                        security_issues.append(f"Dangerous path detected in {key}: {value}")

        # Check for overly permissive permissions (if file paths exist)
        for key, value in config.items():
            if 'path' in key.lower() and isinstance(value, str):
                if os.path.exists(value):
                    try:
                        stat_info = os.stat(value)
                        # Check if world-readable
                        if stat_info.st_mode & 0o004:
                            security_issues.append(f"World-readable file: {value}")
                    except OSError:
                        pass  # Ignore permission errors during validation

        is_secure = len(security_issues) == 0
        return is_secure, security_issues

    def raise_if_invalid(self) -> None:
        """Raise ConfigurationError if any errors were found during validation."""
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(self.errors)
            from .exceptions import ConfigurationError
            raise ConfigurationError("validation", error_msg)


def validate_configuration(config_dir: str = "config", strict: bool = False) -> None:
    """
    Validate all configuration files and log results.
    
    Args:
        config_dir: Path to configuration directory
        strict: Whether to raise exceptions on non-critical issues
        
    Raises:
        ConfigurationError: If critical validation errors are found
    """
    validator = ConfigValidator(strict_mode=strict)
    is_valid, errors, warnings = validator.validate_all(config_dir)
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"Config Warning: {warning}")
    
    # Log or raise errors
    if errors:
        for error in errors:
            logger.error(f"Config Error: {error}")
        validator.raise_if_invalid()
    else:
        logger.info("✓ Configuration validation passed")


# Convenience function for importing
__all__ = ['ConfigValidator', 'validate_configuration']
