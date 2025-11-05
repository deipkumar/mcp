import os
import uuid
import random
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional, Literal

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.fastmcp import FastMCP

# ---------------------------
# Config & Auth
# ---------------------------
API_KEY = os.getenv("MCP_API_KEY", "dev-abc123")  # set in Vercel env

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
        dep = random.randint(800, 12000)  # monthly direct deposit
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
    "aud_seed_home_equity_cross_sell": {
        "name": "Mortgage Clients Without Card",
        "rules": [
            {"field": "has_mortgage", "op": "eq", "value": True},
            {"field": "monthly_deposit", "op": "gte", "value": 5000},
            {"field": "has_credit_card", "op": "eq", "value": False},
        ],
    },
    "aud_seed_student_banking": {
        "name": "College Banking Bundle",
        "rules": [
            {"field": "is_student", "op": "eq", "value": True},
            {"field": "age", "op": "lte", "value": 24},
            {"field": "digital_engagement", "op": "gte", "value": 40},
        ],
    },
    "aud_seed_digital_reengagement": {
        "name": "Digital Re-Engagement",
        "rules": [
            {"field": "tenure_months", "op": "gte", "value": 12},
            {"field": "monthly_deposit", "op": "gte", "value": 2000},
            {"field": "digital_engagement", "op": "lte", "value": 20},
        ],
    },
}

BASE_ACTIVATION_DEFINITIONS = [
    {"id": "act_seed_active_cash_email_web","audience_id": "aud_seed_ca_cash_rewards","destinations": ["dest_email", "dest_web"]},
    {"id": "act_seed_premier_paid","audience_id": "aud_seed_premier_relationship","destinations": ["dest_web", "dest_paid"]},
]

BASE_CAMPAIGN_DEFINITIONS = [
    {"id": "cmp_seed_active_cash_ca","name": "Active Cash Welcome - California","audience_id": "aud_seed_ca_cash_rewards","channel": "email","objective": "credit_card_acquisition"},
    {"id": "cmp_seed_premier_deepening","name": "Premier Relationship Deepening","audience_id": "aud_seed_premier_relationship","channel": "web","objective": "relationship_deepening"},
]

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
    for activation in BASE_ACTIVATION_DEFINITIONS:
        aud_id = activation["audience_id"]
        if aud_id not in AUDIENCES:
            continue
        ACTIVATIONS[activation["id"]] = {
            "id": activation["id"], "audience_id": aud_id, "destinations": activation["destinations"],
            "status": "completed", "completed_at": datetime.now(UTC).isoformat(),
        }
    for campaign in BASE_CAMPAIGN_DEFINITIONS:
        aud_id = campaign["audience_id"]
        if aud_id not in AUDIENCES:
            continue
        audience_size = AUDIENCES[aud_id]["estimated_count"]
        scheduled_for = campaign.get("scheduled_for") or (datetime.now(UTC) + timedelta(minutes=1))
        scheduled_iso = scheduled_for.isoformat() if isinstance(scheduled_for, datetime) else scheduled_for
        CAMPAIGNS[campaign["id"]] = {
            "id": campaign["id"], "name": campaign["name"], "audience_id": aud_id,
            "channel": campaign["channel"], "objective": campaign.get("objective", "credit_card_acquisition"),
            "scheduled_for": scheduled_iso, "status": "delivered", "delivered_at": datetime.now(UTC).isoformat(),
            "metrics": _campaign_metrics_from_audience_size(audience_size, campaign["channel"]),
        }

# ---------------------------
# MCP Server Definition
# ---------------------------
mcp = FastMCP("gradial-audience-demo", streamable_http_path="/")  # endpoints at mount root

@mcp.tool()
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

@mcp.tool()
def list_destinations() -> Dict[str, Any]:
    return {"destinations": DESTINATIONS}

@mcp.tool()
def preview_audience(rules: List[Dict[str, Any]], limit: int = 5) -> Dict[str, Any]:
    rules = normalize_rules(rules)
    matches = apply_rules(CUSTOMERS, rules)
    sample = matches[: max(0, min(limit, len(matches)))]
    return {"estimated_count": len(matches), "sample": sample}

@mcp.tool()
def create_audience(name: str, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    rules = normalize_rules(rules)
    aud_id = f"aud_{uuid.uuid4().hex[:10]}"
    matches = apply_rules(CUSTOMERS, rules)
    AUDIENCES[aud_id] = {
        "id": aud_id, "name": name, "rules": rules,
        "created_at": datetime.now(UTC).isoformat(), "estimated_count": len(matches),
    }
    return {"id": aud_id, "name": name, "estimated_count": len(matches)}

@mcp.tool()
def list_audiences() -> Dict[str, Any]:
    return {"audiences": list(AUDIENCES.values())}

@mcp.tool()
def estimate_audience(audience_id: str) -> Dict[str, Any]:
    a = AUDIENCES.get(audience_id)
    if not a:
        return {"error": f"audience {audience_id} not found"}
    cnt = len(apply_rules(CUSTOMERS, a["rules"]))
    a["estimated_count"] = cnt
    return {"audience_id": audience_id, "estimated_count": cnt}

@mcp.tool()
def activate_audience(audience_id: str, destinations: List[str]) -> Dict[str, Any]:
    if audience_id not in AUDIENCES:
        return {"error": f"audience {audience_id} not found"}
    job_id = f"act_{uuid.uuid4().hex[:10]}"
    ACTIVATIONS[job_id] = {
        "id": job_id, "audience_id": audience_id, "destinations": destinations,
        "status": "completed", "completed_at": datetime.now(UTC).isoformat()
    }
    return {"activation_id": job_id, "status": "completed"}

def _campaign_metrics_from_audience_size(n: int, channel: str) -> Dict[str, Any]:
    base_open = 0.39 if channel == "email" else 0.0
    base_ctr = 0.065 if channel in ("email", "web") else 0.02
    base_apply = 0.011 if channel in ("email", "web") else 0.006
    rnd = random.Random(n + len(channel))
    open_rate = base_open + rnd.uniform(-0.03, 0.03) if channel == "email" else None
    ctr = base_ctr + rnd.uniform(-0.01, 0.01)
    apply_rate = base_apply + rnd.uniform(-0.005, 0.005)
    sent = n
    opens = int(sent * (open_rate or 0))
    clicks = int(sent * ctr)
    applications = int(sent * apply_rate)
    approvals = int(applications * 0.72)
    return {
        "sent": sent, "opens": opens, "clicks": clicks,
        "applications": applications, "approvals": approvals,
        "open_rate": round(open_rate, 4) if open_rate is not None else None,
        "ctr": round(ctr, 4), "apply_rate": round(apply_rate, 4)
    }

_init_base_data()

# ---------------------------
# HTTP Auth Middleware
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
# Build the MCP ASGI app (to be mounted by api/mcp.py)
# ---------------------------
mcp_app = mcp.streamable_http_app()
mcp_app.add_middleware(APIKeyMiddleware)

__all__ = ["mcp_app", "APIKeyMiddleware", "mcp"]

if __name__ == "__main__":
    # Local stdio mode (optional): uvicorn not required for local MCP clients
    mcp.run(transport="stdio")