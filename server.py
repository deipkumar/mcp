#!/usr/bin/env python3
"""HTTP-based MCP server without SSE"""

import os
import uuid
import random
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# ---------------------------
# Config & Auth
# ---------------------------
API_KEY = os.getenv("MCP_API_KEY", "dev-abc123")

# ---------------------------
# Fake Data Generation
# ---------------------------
random.seed(42)

STATES = ["CA", "NY", "TX", "WA", "FL", "IL", "AZ", "NC", "GA", "CO", "OR"]
CHANNELS = ["email", "web", "mobile", "paid"]
DESTINATIONS = [
    {"id": "dest_email", "name": "Email ESP (FakeESP)", "type": "email"},
    {"id": "dest_web", "name": "Web Personalization (FakeTarget)", "type": "web"},
    {"id": "dest_paid", "name": "Paid Media (FakeDSP)", "type": "paid"},
]

def make_customers(n: int = 5000) -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        state = random.choices(STATES, weights=[18,11,13,8,10,8,6,6,6,7,7], k=1)[0]
        age = random.randint(18, 75)
        dep = random.randint(800, 12000)
        credit_score = int(max(480, min(850, random.gauss(690, 60))))
        has_cc = random.random() < 0.55
        has_mortgage = random.random() < 0.32
        is_student = (age < 25) and random.random() < 0.25
        tenure_months = random.randint(1, 240)
        dig_engagement = random.randint(0, 100)
        seg = []
        if dep >= 4000 and not has_cc:
            seg.append("high_deposit_no_cc")
        if has_mortgage and not has_cc:
            seg.append("mortgage_only")
        if state == "CA" and dep >= 4000 and not has_cc:
            seg.append("ca_cc_opportunity")
        out.append({
            "id": f"c_{i:05d}",
            "state": state,
            "age": age,
            "monthly_deposit": dep,
            "credit_score": credit_score,
            "has_credit_card": has_cc,
            "has_mortgage": has_mortgage,
            "is_student": is_student,
            "tenure_months": tenure_months,
            "digital_engagement": dig_engagement,
            "segments": seg
        })
    return out

CUSTOMERS = make_customers()

def apply_rules(customers: List[Dict[str, Any]], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def match_one(c: Dict[str, Any], r: Dict[str, Any]) -> bool:
        field = r.get("field")
        op = r.get("op")
        val = r.get("value")
        if field not in c and op != "exists":
            return False
        v = c.get(field)
        if op == "eq": return v == val
        if op == "neq": return v != val
        if op == "gt": return v > val
        if op == "gte": return v >= val
        if op == "lt": return v < val
        if op == "lte": return v <= val
        if op == "in": return v in (val or [])
        if op == "not_in": return v not in (val or [])
        if op == "contains": return isinstance(v, list) and val in v
        if op == "exists": return (field in c) if val is None else bool(c.get(field)) == bool(val)
        return False

    def matches(c: Dict[str, Any]) -> bool:
        return all(match_one(c, r) for r in rules)

    return [c for c in customers if matches(c)]

# ---------------------------
# In-memory "DB"
# ---------------------------
AUDIENCES: Dict[str, Dict[str, Any]] = {}
CAMPAIGNS: Dict[str, Dict[str, Any]] = {}
ACTIVATIONS: Dict[str, Dict[str, Any]] = {}

BASE_AUDIENCE_DEFINITIONS = {
    "aud_seed_ca_cash_rewards": {
        "name": "Active Cash Prospects - California",
        "rules": [
            {"field": "state", "op": "eq", "value": "CA"},
            {"field": "monthly_deposit", "op": "gte", "value": 4500},
            {"field": "has_credit_card", "op": "eq", "value": False},
        ],
    },
    "aud_seed_premier_relationship": {
        "name": "Premier Relationship Customers",
        "rules": [
            {"field": "monthly_deposit", "op": "gte", "value": 8000},
            {"field": "credit_score", "op": "gte", "value": 740},
            {"field": "tenure_months", "op": "gte", "value": 36},
            {"field": "digital_engagement", "op": "gte", "value": 50},
        ],
    },
}

def normalize_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aliases = {"equals": "eq", "==": "eq", "!=": "neq", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}
    out = []
    for r in rules:
        rr = dict(r)
        if "attribute" in rr and "field" not in rr:
            rr["field"] = rr.pop("attribute")
        if "operator" in rr and "op" not in rr:
            rr["op"] = rr.pop("operator")
        if rr.get("op") in aliases:
            rr["op"] = aliases[rr["op"]]
        v = rr.get("value")
        if isinstance(v, str) and v.lower() in ("true", "false"):
            rr["value"] = v.lower() == "true"
        out.append(rr)
    return out

def _init_base_data():
    if AUDIENCES:
        return
    for aud_id, config in BASE_AUDIENCE_DEFINITIONS.items():
        rules = normalize_rules(config["rules"])
        matches = apply_rules(CUSTOMERS, rules)
        AUDIENCES[aud_id] = {
            "id": aud_id, "name": config["name"], "rules": rules,
            "created_at": datetime.now(UTC).isoformat(), "estimated_count": len(matches),
        }

_init_base_data()

# ---------------------------
# MCP Tool Implementations
# ---------------------------

def list_attributes() -> Dict[str, Any]:
    return {
        "attributes": {
            "state": {"type": "string", "ops": ["eq", "neq", "in", "not_in"], "examples": STATES},
            "age": {"type": "number", "ops": ["gte", "lte", "gt", "lt"]},
            "monthly_deposit": {"type": "number", "ops": ["gte", "lte", "gt", "lt"]},
            "credit_score": {"type": "number", "ops": ["gte", "lte"]},
            "has_credit_card": {"type": "boolean", "ops": ["eq"]},
            "has_mortgage": {"type": "boolean", "ops": ["eq"]},
            "is_student": {"type": "boolean", "ops": ["eq"]},
            "tenure_months": {"type": "number", "ops": ["gte", "lte"]},
            "digital_engagement": {"type": "number", "ops": ["gte", "lte"]},
            "segments": {"type": "array", "ops": ["contains"], "examples": ["high_deposit_no_cc","mortgage_only","ca_cc_opportunity"]},
        }
    }

def list_destinations() -> Dict[str, Any]:
    return {"destinations": DESTINATIONS}

def preview_audience(rules: List[Dict[str, Any]], limit: int = 5) -> Dict[str, Any]:
    rules = normalize_rules(rules)
    matches = apply_rules(CUSTOMERS, rules)
    sample = matches[: max(0, min(limit, len(matches)))]
    return {"estimated_count": len(matches), "sample": sample}

def create_audience(name: str, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    rules = normalize_rules(rules)
    aud_id = f"aud_{uuid.uuid4().hex[:10]}"
    matches = apply_rules(CUSTOMERS, rules)
    AUDIENCES[aud_id] = {
        "id": aud_id, "name": name, "rules": rules,
        "created_at": datetime.now(UTC).isoformat(), "estimated_count": len(matches),
    }
    return {"id": aud_id, "name": name, "estimated_count": len(matches)}

def list_audiences() -> Dict[str, Any]:
    return {"audiences": list(AUDIENCES.values())}

def estimate_audience(audience_id: str) -> Dict[str, Any]:
    a = AUDIENCES.get(audience_id)
    if not a:
        return {"error": f"audience {audience_id} not found"}
    cnt = len(apply_rules(CUSTOMERS, a["rules"]))
    a["estimated_count"] = cnt
    return {"audience_id": audience_id, "estimated_count": cnt}

def activate_audience(audience_id: str, destinations: List[str]) -> Dict[str, Any]:
    if audience_id not in AUDIENCES:
        return {"error": f"audience {audience_id} not found"}
    job_id = f"act_{uuid.uuid4().hex[:10]}"
    ACTIVATIONS[job_id] = {
        "id": job_id, "audience_id": audience_id, "destinations": destinations,
        "status": "completed", "completed_at": datetime.now(UTC).isoformat()
    }
    return {"activation_id": job_id, "status": "completed"}

# Tool registry
TOOLS = {
    "list_attributes": {
        "name": "list_attributes",
        "description": "List all available customer attributes and their operators",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": lambda params: list_attributes()
    },
    "list_destinations": {
        "name": "list_destinations",
        "description": "List all available marketing destinations",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": lambda params: list_destinations()
    },
    "preview_audience": {
        "name": "preview_audience",
        "description": "Preview an audience with given rules",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rules": {"type": "array", "description": "Audience rules"},
                "limit": {"type": "number", "description": "Sample size", "default": 5}
            },
            "required": ["rules"]
        },
        "handler": lambda params: preview_audience(**params)
    },
    "create_audience": {
        "name": "create_audience",
        "description": "Create a new audience",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Audience name"},
                "rules": {"type": "array", "description": "Audience rules"}
            },
            "required": ["name", "rules"]
        },
        "handler": lambda params: create_audience(**params)
    },
    "list_audiences": {
        "name": "list_audiences",
        "description": "List all audiences",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": lambda params: list_audiences()
    },
    "estimate_audience": {
        "name": "estimate_audience",
        "description": "Estimate audience size",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audience_id": {"type": "string", "description": "Audience ID"}
            },
            "required": ["audience_id"]
        },
        "handler": lambda params: estimate_audience(**params)
    },
    "activate_audience": {
        "name": "activate_audience",
        "description": "Activate an audience to destinations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audience_id": {"type": "string", "description": "Audience ID"},
                "destinations": {"type": "array", "description": "Destination IDs"}
            },
            "required": ["audience_id", "destinations"]
        },
        "handler": lambda params: activate_audience(**params)
    },
}

# ---------------------------
# HTTP Endpoints
# ---------------------------

async def handle_mcp_request(request: Request):
    """Handle MCP JSON-RPC requests"""
    try:
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id", 1)

        # Handle MCP protocol methods
        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "gradial-audience-demo",
                        "version": "1.0.0"
                    }
                }
            })

        elif method == "tools/list":
            tools_list = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": tool["inputSchema"]
                }
                for tool in TOOLS.values()
            ]
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": tools_list
                }
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_params = params.get("arguments", {})

            if tool_name not in TOOLS:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool not found: {tool_name}"
                    }
                }, status_code=404)

            tool = TOOLS[tool_name]
            result = tool["handler"](tool_params)

            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result) if not isinstance(result, str) else result
                        }
                    ]
                }
            })

        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }, status_code=404)

    except Exception as e:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }, status_code=500)

# ---------------------------
# Auth Middleware
# ---------------------------
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        key = None
        auth = request.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if key is None:
            key = request.headers.get("x-api-key")
        if key is None:
            key = request.query_params.get("api_key")
        expected = API_KEY
        if expected and key != expected:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

# ---------------------------
# Application
# ---------------------------
app = Starlette(
    debug=True,
    routes=[
        Route("/", handle_mcp_request, methods=["POST"]),
    ],
)

app.add_middleware(APIKeyMiddleware)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
