from typing import Any
from datetime import datetime

def call_display_name(call:Any)->str:
    return f"{call.func_name}__{datetime.now()}"