# tools/datetools.py
from datetime import datetime, timedelta
import calendar
from langchain.tools import Tool

def format_date_for_kb(date):
    """Format date to match the KB format (e.g.g., 'Tue 5/13/25')"""
    day_name = calendar.day_abbr[date.weekday()]
    return f"{day_name} {date.month}/{date.day}/{str(date.year)[2:]}"

def get_current_date():
    """Return the current date in multiple formats"""
    now = datetime.now()
    return {
        "iso_format": now.strftime("%Y-%m-%d"),
        "kb_format": format_date_for_kb(now),
        "verbose_format": now.strftime("%B %d, %Y"),
        "timestamp": now.timestamp(),
        "day_of_week": calendar.day_name[now.weekday()]
    }

def is_date_passed(target_date_str, format="%m/%d/%y"):
    """Check if a specific date has passed compared to today"""
    try:
        target_date = datetime.strptime(target_date_str, format)
        today = datetime.now()
        return target_date < today
    except ValueError:
        return f"Error: Could not parse date '{target_date_str}'. Please use MM/DD/YY format."

def get_date_tools():
    """Create only the date tools needed for the system"""
    return [
        Tool(
            name="GetCurrentDate",
            func=lambda x: str(get_current_date()),
            description="Use this to get the current date in various formats. Call with empty string."
        ),
        Tool(
            name="IsDatePassed",
            func=is_date_passed,
            description="Check if a specific date has passed. Input should be a date in MM/DD/YY format."
        )
    ]