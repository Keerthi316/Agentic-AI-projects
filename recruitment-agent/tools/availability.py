"""
Mock availability tool for interview scheduling.
"""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.schemas import TimeSlot


def check_availability(candidate_name: str) -> list:
    """
    Mock tool that returns available time slots for a candidate.
    """
    random.seed(hash(candidate_name) % 10000)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", "14:00-15:00", "15:00-16:00", "16:00-17:00"]
    
    available = []
    for day in days:
        day_slots = random.sample(time_slots, k=random.randint(2, 4))
        for slot in day_slots:
            start, end = slot.split("-")
            available.append({
                "day": day,
                "start_time": start,
                "end_time": end
            })
    
    return available