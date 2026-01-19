import datetime
import json
import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger("NewsCalendar")

class NewsEvent:
    def __init__(self, title: str, time_str: str, impact: str, currency: str):
        self.title = title
        self.time_str = time_str # Format: "YYYY-MM-DD HH:MM"
        self.impact = impact # "HIGH", "MEDIUM", "LOW"
        self.currency = currency # "USD", "EUR", etc.
        
        try:
            # Try parsing with seconds first, then without
            try:
                self.timestamp = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                self.timestamp = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        except ValueError:
            logger.error(f"Invalid time format for news event: {time_str}")
            self.timestamp = datetime.datetime.now() + datetime.timedelta(days=365) # Push to future

class NewsCalendar:
    """
    Advanced News Calendar 'Event Horizon'.
    Manages high-impact news events and trading blackouts.
    """
    def __init__(self, config_path: str = "config/news_events.json"):
        self.config_path = config_path
        self.events: List[NewsEvent] = []
        self.load_events()
        
    def load_events(self):
        """Load news events from JSON file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"News config not found at {self.config_path}. Creating template.")
            self._create_template()
            return

        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                for item in data.get('events', []):
                    event = NewsEvent(
                        item['title'],
                        item['time'],
                        item['impact'],
                        item['currency']
                    )
                    self.events.append(event)
            logger.info(f"Loaded {len(self.events)} news events.")
        except Exception as e:
            logger.error(f"Failed to load news events: {e}")

    def _create_template(self):
        """Create a template news file."""
        template = {
            "events": [
                {
                    "title": "FOMC Meeting",
                    "time": "2023-12-13 19:00",
                    "impact": "HIGH",
                    "currency": "USD"
                },
                {
                    "title": "NFP",
                    "time": "2023-12-08 13:30",
                    "impact": "HIGH",
                    "currency": "USD"
                }
            ]
        }
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(template, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to create news template: {e}")

    def is_blackout_period(self, symbol: str = "XAUUSD") -> bool:
        """
        Check if we are currently in a news blackout period.
        """
        now = datetime.datetime.now()
        
        for event in self.events:
            # Filter by currency relevance (Simple check)
            # If symbol is XAUUSD, we care about USD news.
            relevant = False
            if "USD" in symbol and event.currency == "USD": relevant = True
            if "EUR" in symbol and event.currency == "EUR": relevant = True
            if "GBP" in symbol and event.currency == "GBP": relevant = True
            if "JPY" in symbol and event.currency == "JPY": relevant = True
            
            if not relevant:
                continue
                
            if event.impact != "HIGH":
                continue
                
            # Define Blackout Window
            # Pre-News: 30 mins before
            # Post-News: 30 mins after
            pre_window = datetime.timedelta(minutes=30)
            post_window = datetime.timedelta(minutes=30)
            
            if (event.timestamp - pre_window) <= now <= (event.timestamp + post_window):
                logger.warning(f"[EVENT HORIZON] Trading Blackout: {event.title} ({event.time_str})")
                return True
                
        return False
