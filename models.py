"""
models.py – Core financial-calculation engine for Studio Feasibility Calculator.

All monetary values are in USD.  Periods are calendar months (1-indexed).
Cost structure mirrors a real boutique fitness studio P&L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────────────
# Data containers
# ────────────────────────────────────────────────────────────────────

@dataclass
class FixedCosts:
    """Monthly fixed operating costs – mirrors a real studio chart of accounts."""
    rent: float = 0.0                       # Rent & Lease
    manager_wages: float = 4_000.0          # Manager Expenses
    manager_payroll_taxes: float = 480.0    # ~12% of manager wages
    owner_wages: float = 0.0                # Owner Wages (optional)
    owner_payroll_taxes: float = 0.0        # Owner Payroll Taxes
    professional_fees: float = 300.0        # Legal / accounting
    insurance: float = 800.0                # General liability + property
    accountable_plan_reimb: float = 0.0     # Accountable Plan Reimbursements
    repairs_maintenance: float = 400.0      # Repairs & Maintenance
    computer_internet: float = 150.0        # Computer & Internet
    telephone: float = 100.0                # Telephone/Internet
    gym_music: float = 50.0                 # Music licensing (e.g. ASCAP/BMI)
    gym_supplies: float = 200.0             # Towels, cleaning, mats, etc.
    office_expenses: float = 100.0          # Office Expenses
    client_processing_software: float = 500.0  # Mindbody, Mariana Tek, etc.
    small_equipment: float = 100.0          # Small Equipment replacement
    automobile_expense: float = 0.0         # Automobile Expense
    security_fee: float = 75.0              # Security / alarm monitoring
    dues_subscriptions: float = 100.0       # Dues & Subscriptions
    bank_service_charges: float = 50.0      # Bank Service Charges
    licenses_permits: float = 50.0          # License & Permits (amortized monthly)
    business_taxes: float = 0.0             # Business Taxes (state/local/property – monthly)
    marketing: float = 3_000.0              # Marketing (Discretionary)

    @property
    def total(self) -> float:
        return (
            self.rent + self.manager_wages + self.manager_payroll_taxes
            + self.owner_wages + self.owner_payroll_taxes
            + self.professional_fees + self.insurance
            + self.accountable_plan_reimb + self.repairs_maintenance
            + self.computer_internet + self.telephone + self.gym_music
            + self.gym_supplies + self.office_expenses
            + self.client_processing_software + self.small_equipment
            + self.automobile_expense + self.security_fee
            + self.dues_subscriptions + self.bank_service_charges
            + self.licenses_permits + self.business_taxes + self.marketing
        )


@dataclass
class VariableCosts:
    """Variable costs that scale with members or revenue."""
    trainer_hourly_rate: float = 35.0       # Trainer Wages (per class hour)
    classes_per_week: int = 30              # Weekly class count
    trainer_payroll_tax_pct: float = 12.0   # Trainer Payroll Taxes %
    processing_fee_pct: float = 2.9         # Firstach / CC processing %

    def monthly_trainer_cost(self) -> float:
        weeks = 4.33
        gross = self.trainer_hourly_rate * self.classes_per_week * weeks
        return round(gross * (1 + self.trainer_payroll_tax_pct / 100.0), 2)

    def monthly_processing_fee(self, revenue: float) -> float:
        return round(revenue * (self.processing_fee_pct / 100.0), 2)

    def total(self, revenue: float) -> float:
        return self.monthly_trainer_cost() + self.monthly_processing_fee(revenue)


@dataclass
class StudioInputs:
    """User-provided + default inputs for the feasibility model."""

    # Location
    city: str = "Seattle"
    state: str = "WA"
    location_label: str = ""                # friendly name for comparisons

    # Physical
    sqft: int = 2_500

    # Revenue
    arpm: float = 199.0                     # avg revenue per member / month (year 1)
    arpm_year2: Optional[float] = None      # ARPM from month 13 onward (None = same as year 1)
    target_members: int = 250
    add_ons_enabled: bool = False
    add_on_revenue_per_member: float = 25.0
    extra_income_enabled: bool = False
    extra_income_monthly: float = 0.0       # flat monthly extra income (PT, retail, events, etc.)

    # Ramp — net new members per month (new joins minus cancellations)
    starting_members: int = 30
    monthly_net_adds: int = 15              # net new members/month (joins - churn)
    months_to_maturity: int = 12

    # Costs
    fixed_costs: FixedCosts = field(default_factory=FixedCosts)
    variable_costs: VariableCosts = field(default_factory=VariableCosts)
    simplified_costs: bool = False              # True = use totals below instead of line items
    simplified_fixed_total: float = 0.0         # user-entered total fixed costs (monthly)
    simplified_variable_total: float = 0.0      # user-entered total variable costs (monthly)

    # Cost change over time (compounding growth/decline)
    fixed_cost_change_pct: float = 0.0          # % change for fixed costs (+/-)
    fixed_cost_change_mode: str = "yearly"      # "monthly" or "yearly"
    variable_cost_change_pct: float = 0.0       # % change for variable costs (+/-)
    variable_cost_change_mode: str = "yearly"   # "monthly" or "yearly"

    # Other expenses (one-time / irregular)
    other_expense_amount: float = 0.0
    other_expense_months: List[int] = field(default_factory=list)  # e.g., [3, 6, 12]

    # Startup
    capex: float = 150_000.0
    opening_costs: float = 15_000.0

    # Studio mode
    studio_mode: str = "new"                # "new" or "existing"

    # Taxes
    ignore_taxes: bool = True
    tax_rate: float = 25.0

    # Scenario
    scenario: str = "base"
    scenario_factors: Dict[str, Dict] = field(default_factory=dict)

    # Rent helper (used when rent=0 → auto-estimate)
    rent_rate_sqft_annual: float = 30.0
    cam_pct: float = 15.0

    @property
    def total_startup(self) -> float:
        # Existing studios have no startup costs (sunk)
        if self.studio_mode == "existing":
            return 0.0
        return self.capex + self.opening_costs

    @property
    def display_label(self) -> str:
        return self.location_label or f"{self.city}, {self.state}"


@dataclass
class MonthRow:
    """Single month of the pro forma."""
    month: int
    members: int
    arpm_used: float
    revenue_membership: float
    revenue_addons: float
    revenue_extra: float
    revenue_total: float
    # Fixed costs (summarized)
    fixed_costs_total: float
    # Variable costs
    trainer_costs: float
    processing_fees: float
    variable_costs_total: float
    # Other expenses (one-time / irregular)
    other_expense: float = 0.0
    # Totals
    total_opex: float = 0.0
    ebitda: float = 0.0
    taxes: float = 0.0
    operating_cf: float = 0.0
    cumulative_cf: float = 0.0


@dataclass
class FeasibilityResult:
    """Full model output."""
    pro_forma: List[MonthRow]
    break_even_members: Optional[int]
    break_even_revenue: Optional[float]
    break_even_month: Optional[int]
    payback_months: Optional[int]
    roi_1y: Optional[float]
    roi_3y: Optional[float]
    irr: Optional[float]
    profit_margin_m24: Optional[float]       # EBITDA / Revenue at month 24 (%)
    monthly_profit_m24: Optional[float]      # EBITDA at month 24 ($)
    ebitda_positive_month: Optional[int]     # first month where monthly EBITDA > 0
    feasibility_score: float
    traffic_light: str
    inputs_used: StudioInputs
    # Cost breakdown for display
    fixed_cost_breakdown: Dict[str, float] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Forecasting functions
# ────────────────────────────────────────────────────────────────────

def _apply_scenario(inputs: StudioInputs) -> StudioInputs:
    """Return a *copy* of inputs with scenario multipliers applied."""
    default_scenarios = {
        "downside": {"member_factor": 0.75, "arpm_factor": 0.90, "cost_factor": 1.10},
        "base":     {"member_factor": 1.0,  "arpm_factor": 1.0,  "cost_factor": 1.0},
        "upside":   {"member_factor": 1.20, "arpm_factor": 1.05, "cost_factor": 0.95},
    }
    factors = inputs.scenario_factors.get(inputs.scenario) or default_scenarios.get(inputs.scenario, default_scenarios["base"])

    import copy
    si = copy.deepcopy(inputs)
    si.target_members = int(si.target_members * factors["member_factor"])
    si.monthly_net_adds = max(1, int(si.monthly_net_adds * factors["member_factor"]))
    si.arpm = round(si.arpm * factors["arpm_factor"], 2)
    if si.arpm_year2 is not None:
        si.arpm_year2 = round(si.arpm_year2 * factors["arpm_factor"], 2)
    # Scale discretionary costs
    si.fixed_costs.marketing = round(si.fixed_costs.marketing * factors["cost_factor"], 2)
    return si


def forecast_members(
    months: int,
    starting: int,
    net_adds: int,
    capacity: int,
) -> List[int]:
    """Month-by-month member count.

    Month 1 = starting members (no growth).
    Month 2+: members grow by net_adds per month.
    Capped at capacity, floored at 0.
    """
    members: List[int] = []
    current = starting
    for m in range(1, months + 1):
        if m == 1:
            # Month 1: starting members, no growth
            current = starting
        else:
            # Month 2+: grow by net adds
            current = current + net_adds
        current = max(0, min(current, capacity))
        members.append(current)
    return members


def _compute_rent(inputs: StudioInputs) -> float:
    """Monthly rent including CAM."""
    if inputs.fixed_costs.rent > 0:
        return inputs.fixed_costs.rent
    base_annual = inputs.rent_rate_sqft_annual * inputs.sqft
    base_monthly = base_annual / 12.0
    cam = base_monthly * (inputs.cam_pct / 100.0)
    return round(base_monthly + cam, 2)


def forecast_financials(inputs: StudioInputs, months: int = 24) -> List[MonthRow]:
    """Build month-by-month pro forma."""
    si = _apply_scenario(inputs)

    member_curve = forecast_members(
        months=months,
        starting=si.starting_members,
        net_adds=si.monthly_net_adds,
        capacity=si.target_members,
    )

    # Ensure rent is set (even in simplified mode, for display purposes)
    rent = _compute_rent(si)
    si.fixed_costs.rent = rent

    trainer_monthly = si.variable_costs.monthly_trainer_cost()

    rows: List[MonthRow] = []
    cumulative_cf = -si.total_startup

    for idx, mem in enumerate(member_curve):
        m = idx + 1
        # ARPM: use year-2 rate from month 13 onward if provided
        current_arpm = si.arpm
        if m > 12 and si.arpm_year2 is not None:
            current_arpm = si.arpm_year2

        rev_mem = round(mem * current_arpm, 2)
        rev_add = round(mem * si.add_on_revenue_per_member, 2) if si.add_ons_enabled else 0.0
        rev_extra = round(si.extra_income_monthly, 2) if si.extra_income_enabled else 0.0
        rev_total = rev_mem + rev_add + rev_extra

        if si.simplified_costs:
            base_fixed = round(si.simplified_fixed_total, 2)
            base_var = round(si.simplified_variable_total, 2)
        else:
            base_fixed = round(si.fixed_costs.total, 2)
            processing = si.variable_costs.monthly_processing_fee(rev_total)
            base_var = round(trainer_monthly + processing, 2)

        # Apply cost compounding (% change over time)
        fixed_total = base_fixed
        if si.fixed_cost_change_pct != 0 and m > 1:
            if si.fixed_cost_change_mode == "monthly":
                factor = (1 + si.fixed_cost_change_pct / 100.0) ** (m - 1)
            else:  # yearly
                factor = (1 + si.fixed_cost_change_pct / 100.0) ** ((m - 1) / 12.0)
            fixed_total = round(base_fixed * factor, 2)

        var_total = base_var
        if si.variable_cost_change_pct != 0 and m > 1:
            if si.variable_cost_change_mode == "monthly":
                factor = (1 + si.variable_cost_change_pct / 100.0) ** (m - 1)
            else:  # yearly
                factor = (1 + si.variable_cost_change_pct / 100.0) ** ((m - 1) / 12.0)
            var_total = round(base_var * factor, 2)

        # Other expense (one-time / irregular)
        other_exp = 0.0
        if si.other_expense_amount > 0 and m in si.other_expense_months:
            other_exp = round(si.other_expense_amount, 2)

        total_opex = round(fixed_total + var_total + other_exp, 2)

        ebitda = round(rev_total - total_opex, 2)

        taxes = 0.0
        if not si.ignore_taxes and ebitda > 0:
            taxes = round(ebitda * (si.tax_rate / 100.0), 2)

        op_cf = round(ebitda - taxes, 2)
        cumulative_cf = round(cumulative_cf + op_cf, 2)

        rows.append(MonthRow(
            month=m,
            members=mem,
            arpm_used=current_arpm,
            revenue_membership=rev_mem,
            revenue_addons=rev_add,
            revenue_extra=rev_extra,
            revenue_total=rev_total,
            fixed_costs_total=fixed_total,
            trainer_costs=trainer_monthly if not si.simplified_costs else 0.0,
            processing_fees=processing if not si.simplified_costs else 0.0,
            variable_costs_total=var_total,
            other_expense=other_exp,
            total_opex=total_opex,
            ebitda=ebitda,
            taxes=taxes,
            operating_cf=op_cf,
            cumulative_cf=cumulative_cf,
        ))

    return rows


def _get_fixed_cost_breakdown(inputs: StudioInputs) -> Dict[str, float]:
    """Return a dict of fixed cost line items for display."""
    fc = inputs.fixed_costs
    return {
        "Rent & Lease": fc.rent,
        "Manager Wages": fc.manager_wages,
        "Manager Payroll Taxes": fc.manager_payroll_taxes,
        "Owner Wages": fc.owner_wages,
        "Owner Payroll Taxes": fc.owner_payroll_taxes,
        "Professional Fees": fc.professional_fees,
        "Insurance": fc.insurance,
        "Accountable Plan Reimb.": fc.accountable_plan_reimb,
        "Repairs & Maintenance": fc.repairs_maintenance,
        "Computer & Internet": fc.computer_internet,
        "Telephone": fc.telephone,
        "Gym Music": fc.gym_music,
        "Gym Supplies": fc.gym_supplies,
        "Office Expenses": fc.office_expenses,
        "Client Processing Software": fc.client_processing_software,
        "Small Equipment": fc.small_equipment,
        "Automobile Expense": fc.automobile_expense,
        "Security Fee": fc.security_fee,
        "Dues & Subscriptions": fc.dues_subscriptions,
        "Bank Service Charges": fc.bank_service_charges,
        "Licenses & Permits": fc.licenses_permits,
        "Business Taxes": fc.business_taxes,
        "Marketing": fc.marketing,
    }


# ────────────────────────────────────────────────────────────────────
# Break-even
# ────────────────────────────────────────────────────────────────────

def break_even(inputs: StudioInputs) -> Tuple[Optional[int], Optional[float], Optional[int]]:
    """Returns (break_even_members, break_even_revenue, break_even_month)."""
    si = _apply_scenario(inputs)
    si.fixed_costs.rent = _compute_rent(si)

    extra_monthly = si.extra_income_monthly if si.extra_income_enabled else 0.0

    if si.simplified_costs:
        total_fixed_equiv = si.simplified_fixed_total + si.simplified_variable_total - extra_monthly
        effective_arpm = si.arpm
        if si.add_ons_enabled:
            effective_arpm += si.add_on_revenue_per_member
        effective_arpm_net = effective_arpm
    else:
        fixed_total = si.fixed_costs.total
        trainer_cost = si.variable_costs.monthly_trainer_cost()
        total_fixed_equiv = fixed_total + trainer_cost - extra_monthly
        effective_arpm = si.arpm
        if si.add_ons_enabled:
            effective_arpm += si.add_on_revenue_per_member
        effective_arpm_net = effective_arpm * (1 - si.variable_costs.processing_fee_pct / 100.0)

    if effective_arpm_net <= 0:
        return (None, None, None)

    be_members = max(0, math.ceil(total_fixed_equiv / effective_arpm_net))
    be_revenue = round(max(0, total_fixed_equiv) / (1 - si.variable_costs.processing_fee_pct / 100.0), 2)

    # Break-even month from pro forma (search up to 10 years for high-startup studios)
    rows = forecast_financials(inputs, months=120)
    be_month: Optional[int] = None
    for r in rows:
        if r.cumulative_cf >= 0:
            be_month = r.month
            break

    # If not found within 120 months but the studio IS profitable at maturity,
    # estimate when cumulative CF would cross zero based on the mature profit rate.
    if be_month is None and len(rows) >= 24:
        last = rows[-1]
        if last.operating_cf > 0 and last.cumulative_cf < 0:
            remaining = abs(last.cumulative_cf)
            extra_months = math.ceil(remaining / last.operating_cf)
            be_month = len(rows) + extra_months

    return (be_members, be_revenue, be_month)


# ────────────────────────────────────────────────────────────────────
# Payback
# ────────────────────────────────────────────────────────────────────

def payback(inputs: StudioInputs) -> Optional[int]:
    """Months to recover total startup cost.

    If total_startup is 0 (existing studio, sunk costs), payback is 0.
    """
    si = _apply_scenario(inputs)
    if si.total_startup == 0:
        return 0
    rows = forecast_financials(inputs, months=120)
    for r in rows:
        if r.cumulative_cf >= 0:
            return r.month

    # If not found within 120 months but the studio IS profitable at maturity,
    # estimate based on the mature monthly profit rate.
    if len(rows) >= 24:
        last = rows[-1]
        if last.operating_cf > 0 and last.cumulative_cf < 0:
            remaining = abs(last.cumulative_cf)
            extra_months = math.ceil(remaining / last.operating_cf)
            return len(rows) + extra_months

    return None


# ────────────────────────────────────────────────────────────────────
# ROI + IRR
# ────────────────────────────────────────────────────────────────────

def roi_metrics(inputs: StudioInputs) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (roi_1y_pct, roi_3y_pct, irr_monthly_annualised).

    When total_startup is 0 (e.g. an existing studio with sunk costs),
    ROI is calculated as cumulative operating profit / total annual opex
    (i.e. return-on-operating-cost), so the metric is still meaningful.
    IRR is skipped (requires an initial outlay).
    """
    si = _apply_scenario(inputs)
    rows = forecast_financials(inputs, months=36)
    startup = si.total_startup

    cum_12 = rows[11].cumulative_cf if len(rows) >= 12 else None
    cum_36 = rows[35].cumulative_cf if len(rows) >= 36 else None

    if startup == 0:
        annual_opex = sum(r.total_opex for r in rows[:12]) if len(rows) >= 12 else None
        if annual_opex and annual_opex > 0:
            roi_1y = round((cum_12 / annual_opex) * 100, 2) if cum_12 is not None else None
            total_opex_3y = sum(r.total_opex for r in rows[:36]) if len(rows) >= 36 else None
            roi_3y = round((cum_36 / total_opex_3y) * 100, 2) if cum_36 is not None and total_opex_3y else None
        else:
            roi_1y = None
            roi_3y = None
        return (roi_1y, roi_3y, None)

    roi_1y = round((cum_12 / startup) * 100, 2) if cum_12 is not None else None
    roi_3y = round((cum_36 / startup) * 100, 2) if cum_36 is not None else None

    irr_val: Optional[float] = None
    try:
        cfs = [-startup] + [r.operating_cf for r in rows]
        monthly_irr = _irr_bisect(cfs)
        if monthly_irr is not None and not math.isnan(monthly_irr):
            irr_val = round(((1 + monthly_irr) ** 12 - 1) * 100, 2)
    except Exception:
        pass

    return (roi_1y, roi_3y, irr_val)


def _irr_bisect(cashflows: List[float], tol: float = 1e-6, max_iter: int = 2000) -> Optional[float]:
    """Simple bisection IRR solver for monthly cash flows."""
    def npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))

    brackets = [(-0.5, 2.0), (-0.9, 5.0), (-0.99, 10.0)]
    lo, hi = -0.5, 2.0
    found = False
    for _lo, _hi in brackets:
        try:
            if npv(_lo) * npv(_hi) <= 0:
                lo, hi = _lo, _hi
                found = True
                break
        except (ZeroDivisionError, OverflowError):
            continue
    if not found:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        try:
            val = npv(mid)
        except (ZeroDivisionError, OverflowError):
            return None
        if abs(val) < tol:
            return mid
        try:
            if npv(lo) * val < 0:
                hi = mid
            else:
                lo = mid
        except (ZeroDivisionError, OverflowError):
            return None
    return (lo + hi) / 2.0


# ────────────────────────────────────────────────────────────────────
# Sensitivity (one-way)
# ────────────────────────────────────────────────────────────────────

def sensitivity(
    inputs: StudioInputs,
    param: str,
    values: List[float],
) -> List[Dict]:
    """Run model for each value of *param*, return list of metric dicts."""
    import copy
    results = []
    for v in values:
        si = copy.deepcopy(inputs)
        if param == "rent":
            si.fixed_costs.rent = v
        elif param == "marketing":
            si.fixed_costs.marketing = v
        elif param == "trainer_hourly_rate":
            si.variable_costs.trainer_hourly_rate = v
        else:
            if hasattr(si, param):
                setattr(si, param, v)
        rows = forecast_financials(si, months=24)
        _, _, be_month = break_even(si)
        pb = payback(si)
        roi_1, roi_3, irr = roi_metrics(si)
        results.append({
            "param": param,
            "value": v,
            "ebitda_m24": rows[-1].ebitda if rows else None,
            "cum_cf_m24": rows[-1].cumulative_cf if rows else None,
            "break_even_month": be_month,
            "payback_months": pb,
            "roi_1y": roi_1,
        })
    return results


# ────────────────────────────────────────────────────────────────────
# Go / No-Go scoring
# ────────────────────────────────────────────────────────────────────

def feasibility_score(
    ebitda_positive_month: Optional[int],
    profit_margin_m24: Optional[float],
    pb_months: Optional[int],
    roi_1y: Optional[float],
    thresholds: Optional[Dict] = None,
) -> Tuple[float, str]:
    """Returns (score 0-100, traffic_light).

    Four equally weighted components (25 pts each):
      1. Operating Break-Even: month when monthly EBITDA first goes positive
      2. Profit Margin at Month 24: EBITDA / Revenue %
      3. Payback Period: months to recover startup investment
      4. 1-Year ROI: cumulative CF / startup investment
    """
    t = thresholds or {
        "ebitda_month_green": 10, "ebitda_month_yellow": 18,
        "margin_green": 15.0, "margin_yellow": 0.0,
        "payback_green": 24, "payback_yellow": 36,
        "roi_1y_green": 0.0, "roi_1y_yellow": -20.0,
    }
    score = 0.0

    # 1. Operating break-even (lower is better)
    if ebitda_positive_month is not None:
        if ebitda_positive_month <= t["ebitda_month_green"]:
            score += 25.0
        elif ebitda_positive_month <= t["ebitda_month_yellow"]:
            score += 15.0
        else:
            score += 5.0

    # 2. Profit margin at month 24 (higher is better)
    if profit_margin_m24 is not None:
        if profit_margin_m24 >= t["margin_green"]:
            score += 25.0
        elif profit_margin_m24 >= t["margin_yellow"]:
            score += 15.0
        else:
            score += 5.0

    # 3. Payback period (lower is better)
    if pb_months is not None:
        if pb_months <= t["payback_green"]:
            score += 25.0
        elif pb_months <= t["payback_yellow"]:
            score += 15.0
        else:
            score += 5.0

    # 4. 1-Year ROI (higher is better)
    if roi_1y is not None:
        if roi_1y >= t["roi_1y_green"]:
            score += 25.0
        elif roi_1y >= t["roi_1y_yellow"]:
            score += 15.0
        else:
            score += 5.0

    score = round(min(score, 100.0), 1)
    if score >= 70:
        light = "green"
    elif score >= 40:
        light = "yellow"
    else:
        light = "red"
    return (score, light)


# ────────────────────────────────────────────────────────────────────
# Main run
# ────────────────────────────────────────────────────────────────────

def run_model(inputs: StudioInputs, months: int = 24, thresholds: Optional[Dict] = None) -> FeasibilityResult:
    """Orchestrate all calculations and return a FeasibilityResult."""
    import copy
    si = copy.deepcopy(inputs)
    si.fixed_costs.rent = _compute_rent(si)
    inputs.fixed_costs.rent = si.fixed_costs.rent  # persist back

    rows = forecast_financials(inputs, months=months)
    be_members, be_revenue, be_month = break_even(inputs)
    pb = payback(inputs)
    roi_1, roi_3, irr = roi_metrics(inputs)
    breakdown = _get_fixed_cost_breakdown(inputs)

    # Profit margin at month 24 (or last month): EBITDA / Revenue
    last = rows[-1] if rows else None
    if last and last.revenue_total > 0:
        margin_m24 = round((last.ebitda / last.revenue_total) * 100, 1)
        profit_m24 = round(last.ebitda, 2)
    else:
        margin_m24 = None
        profit_m24 = None

    # Operating break-even: first month where monthly EBITDA > 0
    ebitda_pos = None
    for r in rows:
        if r.ebitda > 0:
            ebitda_pos = r.month
            break

    score, light = feasibility_score(ebitda_pos, margin_m24, pb, roi_1, thresholds)

    return FeasibilityResult(
        pro_forma=rows,
        break_even_members=be_members,
        break_even_revenue=be_revenue,
        break_even_month=be_month,
        payback_months=pb,
        roi_1y=roi_1,
        roi_3y=roi_3,
        irr=irr,
        profit_margin_m24=margin_m24,
        monthly_profit_m24=profit_m24,
        ebitda_positive_month=ebitda_pos,
        feasibility_score=score,
        traffic_light=light,
        inputs_used=inputs,
        fixed_cost_breakdown=breakdown,
    )
