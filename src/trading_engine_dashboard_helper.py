    def _print_dashboard(self, symbol_positions, bucket_pnl, current_rsi, current_atr, pressure_metrics, force=False):
        """Print the Aether Intelligence Dashboard."""
        current_time = time.time()
        
        # Determine Strategy Status
        strategy_status = "Monitoring"
        if bucket_pnl > 0:
            strategy_status = "PROFIT SEEKING (Trailing)"
        elif len(symbol_positions) > 1:
            strategy_status = "ZONE RECOVERY (Hedging)"

        # Initialize tracking vars
        if not hasattr(self, '_last_log_pnl'): self._last_log_pnl = 0.0
        if not hasattr(self, '_last_log_strategy'): self._last_log_strategy = ""
        if not hasattr(self, '_last_position_status_time'): self._last_position_status_time = 0.0

        # Smart Interval Checks
        is_time_update = (current_time - self._last_position_status_time >= 60.0)
        pnl_change = abs(bucket_pnl - self._last_log_pnl)
        is_pnl_significant = (pnl_change > 5.0) # $5 change
        is_strategy_change = (strategy_status != self._last_log_strategy)
        
        if force or is_time_update or is_pnl_significant or is_strategy_change:
            self._last_position_status_time = current_time
            self._last_log_pnl = bucket_pnl
            self._last_log_strategy = strategy_status
            
            # Format PnL
            pnl_symbol = "+" if bucket_pnl >= 0 else ""
            pnl_str = f"{pnl_symbol}${bucket_pnl:.2f}"
            
            # Format Pressure
            pressure_str = "BALANCED"
            if pressure_metrics:
                dom = pressure_metrics.get('dominance', 'NEUTRAL')
                intensity = pressure_metrics.get('intensity', 'LOW')
                pressure_str = f"{dom} ({intensity})"

            rsi_disp = "NA" if current_rsi is None else f"{float(current_rsi):.1f}"
            atr_disp = "NA" if current_atr is None else f"{float(current_atr):.4f}"
            
            # Clean Dashboard Log
            status_msg = (
                f"\n================ [ AETHER INTELLIGENCE DASHBOARD ] ================\n"
                f" STATUS:       {strategy_status}\n"
                f" POSITIONS:    {len(symbol_positions)} Active | PnL: {pnl_str}\n"
                f" MARKET:       RSI {rsi_disp} | ATR {atr_disp} | Pressure: {pressure_str}\n"
                f" NEXT ACTION:  Holding & Analyzing Tick Data... (Supreme IO)\n"
                f"==================================================================="
            )
            
            import sys
            try:
                    ui_logger.info(status_msg)
            except Exception:
                    print(status_msg, flush=True)

