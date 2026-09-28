"""
streamlit_app.py – Studio Feasibility Calculator dashboard (v7).

Section order (strictly enforced):
  1. Key Metrics (both locations side-by-side in compare mode)
  2. Feasibility Score + Go/No-Go explanation
  3. Market Research (Census Reporter via FIPS)
  4. Expense Tracker (Fixed / Variable / Add-On / Other)
  5. 24-Month Projections
  6. Sensitivity Analysis
  7. Full Pro Forma Table + Exports

Key changes in v6:
  - Removed churn — members grow linearly by net adds until target cap
  - Studio mode: New Studio vs Existing Studio (sunk costs)
  - Add-on expenses with compounding (monthly/yearly)
  - Other expenses (one-time / irregular by month)
  - Competitor search radius slider (0.2–10 mi)
  - Compare mode: separate settings per location, single Compare button
  - No shared settings in compare mode

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from market_research import MarketData, research_location
from models import (
    FeasibilityResult,
    FixedCosts,
    MonthRow,
    StudioInputs,
    VariableCosts,
    break_even,
    feasibility_score,
    forecast_financials,
    payback,
    roi_metrics,
    run_model,
    sensitivity,
)

# ────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Studio Feasibility Calculator",
    page_icon="\U0001f3cb\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────
# Logo banner
# ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:#1a1a2e; border-radius:12px; padding:18px 32px; margin-bottom:1.5rem;
            display:flex; align-items:center; justify-content:center;">
    <span style="color:#ffffff; font-size:1.4rem; font-weight:700; letter-spacing:0.02em;">
        Studio Feasibility Calculator
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.mc { background:#fff; border:1px solid #e2e8f0; border-radius:12px;
      padding:1rem 0.75rem; text-align:center; min-height:120px;
      display:flex; flex-direction:column; justify-content:center;
      box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.mc:hover { box-shadow:0 4px 12px rgba(0,0,0,0.08); }
.mc .ic { font-size:1.3rem; margin-bottom:4px; }
.mc .lb { font-size:0.68rem; color:#64748b; font-weight:600;
           text-transform:uppercase; letter-spacing:0.05em; }
.mc .vl { font-size:1.4rem; font-weight:700; color:#1e293b; margin:2px 0; }
.mc .sb { font-size:0.66rem; color:#94a3b8; }

.tl { display:inline-block; padding:8px 24px; border-radius:999px;
      font-weight:700; font-size:0.9rem; }
.tl-green  { background:#dcfce7; color:#166534; }
.tl-yellow { background:#fef9c3; color:#854d0e; }
.tl-red    { background:#fee2e2; color:#991b1b; }

.cmp-winner { background:#dcfce7; border:2px solid #22c55e; border-radius:12px; padding:0.5rem; }
.cmp-loser  { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:0.5rem; }

.cost-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.cost-table th { text-align:left; padding:8px 6px; border-bottom:2px solid #e2e8f0;
                 color:#475569; font-weight:700; font-size:0.75rem;
                 text-transform:uppercase; letter-spacing:0.04em; }
.cost-table td { padding:6px; border-bottom:1px solid #f1f5f9; }
.cost-table .cost-icon { width:20px; text-align:center; }
.cost-table .cost-name { color:#334155; }
.cost-table .cost-val  { text-align:right; font-weight:600; color:#1e293b; font-variant-numeric:tabular-nums; }
.cost-table .cost-total td { border-top:2px solid #e2e8f0; font-weight:700; color:#0f172a; }

.score-explain { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                 padding:1.25rem; margin:0.5rem 0; }
.score-explain h4 { margin:0 0 0.75rem 0; color:#0f172a; font-size:1rem; }
.score-bar { display:flex; align-items:center; margin:6px 0; }
.score-bar-label { width:180px; font-size:0.82rem; color:#475569; }
.score-bar-track { flex:1; height:8px; background:#e2e8f0; border-radius:4px; overflow:hidden; }
.score-bar-fill  { height:100%; border-radius:4px; }
.score-bar-pts   { width:50px; text-align:right; font-size:0.82rem; font-weight:600; color:#1e293b; }

.src-badge { display:inline-block; padding:3px 10px; border-radius:999px;
             font-size:0.72rem; font-weight:600; letter-spacing:0.03em; }
.src-live { background:#dcfce7; color:#166534; }
.src-est  { background:#fef9c3; color:#854d0e; }

.nbhood-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
               padding:0.75rem 1rem; margin:0.25rem 0; }
.nbhood-score { font-weight:700; font-size:1.1rem; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
# Load config
# ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config_defaults.yaml"

@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CFG = load_config()

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def mc(icon: str, label: str, value: str, sub: str = "") -> str:
    return (f'<div class="mc"><span class="ic">{icon}</span>'
            f'<span class="lb">{label}</span>'
            f'<span class="vl">{value}</span>'
            f'<span class="sb">{sub}</span></div>')

def tl_badge(light: str, score: float) -> str:
    lbl = {"green":"GO","yellow":"CAUTION","red":"NO-GO"}.get(light,"?")
    return f'<span class="tl tl-{light}">{lbl} &#x2022; {score:.0f}/100</span>'

def source_badge(market: MarketData) -> str:
    if market.is_live:
        return '<span class="src-badge src-live">&#x1F7E2; Live Census Data</span>'
    return '<span class="src-badge src-est">&#x1F7E1; Estimated Data</span>'

def fd(v) -> str:
    return f"${v:,.0f}" if v is not None else "N/A"

def fp(v) -> str:
    return f"{v:,.1f}%" if v is not None else "N/A"

def fm(v) -> str:
    return f"{v} mo" if v is not None else "N/A"

def _cost_row(icon: str, name: str, val: float) -> str:
    return (f'<tr><td class="cost-icon">{icon}</td>'
            f'<td class="cost-name">{name}</td>'
            f'<td class="cost-val">${val:,.0f}</td></tr>')

def _cost_total_row(label: str, val: float) -> str:
    return (f'<tr class="cost-total"><td></td>'
            f'<td>{label}</td>'
            f'<td class="cost-val">${val:,.0f}</td></tr>')


# ────────────────────────────────────────────────────────────────────
# Sidebar: collect inputs
# ────────────────────────────────────────────────────────────────────

def _collect_settings(prefix: str, defaults: dict) -> dict:
    """Collect all non-location settings (revenue, ramp, costs, etc.).
    Returns dict of all settings values.
    """
    d = defaults
    vals = {}

    # ── Studio Mode ──
    studio_mode_label = st.radio(
        "Studio Type", ["New Studio", "Existing Studio"],
        key=f"{prefix}_studio_mode", horizontal=True,
        help="Existing Studio = startup costs are sunk (already paid).")
    vals["studio_mode"] = "new" if studio_mode_label == "New Studio" else "existing"

    with st.expander("Revenue & Ramp"):
        vals["sqft"] = st.number_input("Studio size (sq ft)", 500, 20000, d.get("sqft", 2500),
                                       step=100, key=f"{prefix}_sqft")
        vals["target"] = st.number_input("Target members", 50, 2000, d.get("target", 250),
                                         step=10, key=f"{prefix}_target")
        vals["scenario"] = st.selectbox("Scenario", ["Base", "Downside", "Upside"],
                                        key=f"{prefix}_scen")
        vals["starting"] = st.number_input("Starting members (pre-sale)", 0, 500,
                                           d.get("starting", 30), step=5, key=f"{prefix}_start")
        vals["net_adds"] = st.number_input(
            "Monthly Net New Members", 1, 200,
            d.get("net_adds", 15), step=1, key=f"{prefix}_adds",
            help="Net growth per month = new sign-ups minus cancellations. "
                 "Example: if 20 join and 5 cancel, net adds = 15.")
        arpm_y2_on = st.toggle("ARPM changes in Year 2", value=d.get("arpm_y2_on", False),
                               key=f"{prefix}_arpm_y2_on")
        vals["arpm_y2"] = None
        if arpm_y2_on:
            vals["arpm_y2"] = st.number_input("ARPM Year 2 ($)", 50.0, 800.0,
                                              d.get("arpm_y2", 199.0), step=5.0,
                                              format="%.2f", key=f"{prefix}_arpm_y2")
        vals["addons"] = st.toggle("Enable add-on revenue", value=False, key=f"{prefix}_addons")
        vals["addon_rev"] = 25.0
        if vals["addons"]:
            vals["addon_rev"] = st.number_input("Add-on $/member/mo", 0.0, 200.0, 25.0,
                                                step=5.0, format="%.0f", key=f"{prefix}_arev")
        vals["extra_income_on"] = st.toggle("Extra monthly income",
                                            value=False, key=f"{prefix}_extra_on",
                                            help="Flat monthly income from PT, retail, events, etc.")
        vals["extra_income"] = 0.0
        if vals["extra_income_on"]:
            vals["extra_income"] = st.number_input(
                "Extra income ($/mo)", 0.0, 100_000.0, 0.0,
                step=250.0, format="%.0f", key=f"{prefix}_extra_amt",
                help="Total additional monthly revenue (personal training, retail, rentals, etc.)")

    with st.expander("Startup Costs"):
        if vals["studio_mode"] == "existing":
            st.info("Existing studio — startup costs are treated as sunk (already paid).")
            vals["capex"] = st.number_input("Build-out + equipment ($)", 0.0, 2_000_000.0,
                                            0.0, step=5000.0,
                                            format="%.0f", key=f"{prefix}_capex",
                                            disabled=True)
            vals["opening"] = st.number_input("Opening costs ($)", 0.0, 200_000.0,
                                              0.0, step=1000.0,
                                              format="%.0f", key=f"{prefix}_opening",
                                              disabled=True)
        else:
            vals["capex"] = st.number_input("Build-out + equipment ($)", 0.0, 2_000_000.0,
                                            d.get("capex", 150_000.0), step=5000.0,
                                            format="%.0f", key=f"{prefix}_capex")
            vals["opening"] = st.number_input("Opening costs ($)", 0.0, 200_000.0,
                                              d.get("opening", 15_000.0), step=1000.0,
                                              format="%.0f", key=f"{prefix}_opening")

    # ── Cost entry mode selector ──
    cost_mode = st.radio("Cost entry mode",
                         ["Simplified", "Detailed"],
                         key=f"{prefix}_cost_mode", horizontal=True,
                         help="Simplified = enter totals only. Detailed = line-by-line.")
    vals["simplified_costs"] = (cost_mode == "Simplified")

    if vals["simplified_costs"]:
        st.markdown(
            '<div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px; '
            'padding:0.75rem; margin-bottom:0.5rem; font-size:0.82rem; color:#0c4a6e;">'
            '<strong>Fixed Costs</strong> include: rent, wages, payroll taxes, insurance, '
            'marketing, utilities, software, repairs, supplies, licenses, and all other '
            'recurring monthly expenses that do not change with revenue.<br><br>'
            '<strong>Variable Costs</strong> include: trainer wages + payroll taxes, '
            'and credit card processing fees — costs that scale with classes or revenue.'
            '</div>', unsafe_allow_html=True)
        vals["simple_fixed"] = st.number_input(
            "Total Fixed Costs ($/mo)", 0.0, 500_000.0, d.get("simple_fixed", 10_000.0),
            step=500.0, format="%.0f", key=f"{prefix}_sfix")
        fc_col1, fc_col2 = st.columns(2)
        with fc_col1:
            vals["fixed_change_pct"] = st.number_input(
                "Fixed cost change (%)", -50.0, 100.0, 0.0, step=0.5, format="%.1f",
                key=f"{prefix}_sfcp", help="Positive = costs rise; negative = costs decline.")
        with fc_col2:
            vals["fixed_change_mode"] = st.radio(
                "Fixed compounding", ["Monthly", "Yearly"],
                key=f"{prefix}_sfcm", horizontal=True).lower()
        vals["simple_variable"] = st.number_input(
            "Total Variable Costs ($/mo)", 0.0, 200_000.0, d.get("simple_variable", 5_000.0),
            step=250.0, format="%.0f", key=f"{prefix}_svar")
        vc_col1, vc_col2 = st.columns(2)
        with vc_col1:
            vals["variable_change_pct"] = st.number_input(
                "Variable cost change (%)", -50.0, 100.0, 0.0, step=0.5, format="%.1f",
                key=f"{prefix}_svcp", help="Positive = costs rise; negative = costs decline.")
        with vc_col2:
            vals["variable_change_mode"] = st.radio(
                "Variable compounding", ["Monthly", "Yearly"],
                key=f"{prefix}_svcm", horizontal=True).lower()
        vals["mgr_wages"] = 0.0; vals["mgr_tax"] = 0.0
        vals["owner_wages"] = 0.0; vals["owner_tax"] = 0.0
        vals["prof_fees"] = 0.0; vals["ins"] = 0.0; vals["acct_reimb"] = 0.0
        vals["comp_int"] = 0.0; vals["phone"] = 0.0; vals["music"] = 0.0
        vals["office"] = 0.0; vals["sw"] = 0.0; vals["auto"] = 0.0
        vals["sec"] = 0.0; vals["dues"] = 0.0; vals["bank"] = 0.0
        vals["lic"] = 0.0; vals["mktg"] = 0.0; vals["repairs"] = 0.0
        vals["supplies"] = 0.0; vals["sm_equip"] = 0.0
        vals["trainer_rate"] = 35.0; vals["classes_wk"] = 30
        vals["trainer_tax"] = 12.0; vals["proc_fee"] = 2.9
    else:
        vals["simple_fixed"] = 0.0
        vals["simple_variable"] = 0.0

        # Defaults — will be overridden inside expanders if user enters values
        if "fixed_change_pct" not in vals:
            vals["fixed_change_pct"] = 0.0
            vals["fixed_change_mode"] = "yearly"
        if "variable_change_pct" not in vals:
            vals["variable_change_pct"] = 0.0
            vals["variable_change_mode"] = "yearly"

        with st.expander("Fixed Costs (monthly)"):
            vals["mgr_wages"] = st.number_input("Manager Wages", 0.0, 30_000.0, 4_000.0,
                                                step=250.0, format="%.0f", key=f"{prefix}_mgr_w")
            vals["mgr_tax"] = st.number_input("Manager Payroll Taxes", 0.0, 10_000.0, 480.0,
                                              step=50.0, format="%.0f", key=f"{prefix}_mgr_t")
            vals["owner_wages"] = st.number_input("Owner Wages", 0.0, 50_000.0, 0.0,
                                                  step=500.0, format="%.0f", key=f"{prefix}_own_w")
            vals["owner_tax"] = st.number_input("Owner Payroll Taxes", 0.0, 15_000.0, 0.0,
                                                step=50.0, format="%.0f", key=f"{prefix}_own_t")
            vals["prof_fees"] = st.number_input("Professional Fees", 0.0, 5_000.0, 300.0,
                                                step=50.0, format="%.0f", key=f"{prefix}_prof")
            vals["ins"] = st.number_input("Insurance", 0.0, 10_000.0, 800.0,
                                          step=100.0, format="%.0f", key=f"{prefix}_ins")
            vals["acct_reimb"] = st.number_input("Accountable Plan Reimb.", 0.0, 5_000.0, 0.0,
                                                 step=50.0, format="%.0f", key=f"{prefix}_acct")
            vals["comp_int"] = st.number_input("Computer & Internet", 0.0, 2_000.0, 150.0,
                                               step=25.0, format="%.0f", key=f"{prefix}_comp")
            vals["phone"] = st.number_input("Telephone", 0.0, 1_000.0, 100.0,
                                            step=25.0, format="%.0f", key=f"{prefix}_phone")
            vals["music"] = st.number_input("Gym Music License", 0.0, 500.0, 50.0,
                                            step=10.0, format="%.0f", key=f"{prefix}_music")
            vals["office"] = st.number_input("Office Expenses", 0.0, 1_000.0, 100.0,
                                             step=25.0, format="%.0f", key=f"{prefix}_offc")
            vals["sw"] = st.number_input("Client Processing Software", 0.0, 5_000.0, 500.0,
                                         step=50.0, format="%.0f", key=f"{prefix}_sw")
            vals["auto"] = st.number_input("Automobile Expense", 0.0, 2_000.0, 0.0,
                                           step=50.0, format="%.0f", key=f"{prefix}_auto")
            vals["sec"] = st.number_input("Security Fee", 0.0, 1_000.0, 75.0,
                                          step=25.0, format="%.0f", key=f"{prefix}_sec")
            vals["dues"] = st.number_input("Dues & Subscriptions", 0.0, 1_000.0, 100.0,
                                           step=25.0, format="%.0f", key=f"{prefix}_dues")
            vals["bank"] = st.number_input("Bank Service Charges", 0.0, 500.0, 50.0,
                                           step=10.0, format="%.0f", key=f"{prefix}_bank")
            vals["lic"] = st.number_input("Licenses & Permits", 0.0, 1_000.0, 50.0,
                                          step=10.0, format="%.0f", key=f"{prefix}_lic")
            vals["biz_taxes"] = st.number_input("Business Taxes (monthly)", 0.0, 5_000.0, 0.0,
                                                step=50.0, format="%.0f", key=f"{prefix}_btax",
                                                help="State, local, or property taxes — enter monthly amount.")
            vals["mktg"] = st.number_input("Marketing", 0.0, 50_000.0, d.get("marketing", 3_000.0),
                                           step=250.0, format="%.0f", key=f"{prefix}_mktg")
            vals["repairs"] = st.number_input("Repairs & Maintenance", 0.0, 5_000.0, 400.0,
                                              step=50.0, format="%.0f", key=f"{prefix}_rep")
            vals["supplies"] = st.number_input("Gym Supplies", 0.0, 2_000.0, 200.0,
                                               step=25.0, format="%.0f", key=f"{prefix}_supp")
            vals["sm_equip"] = st.number_input("Small Equipment", 0.0, 2_000.0, 100.0,
                                               step=25.0, format="%.0f", key=f"{prefix}_smeq")
            st.markdown("---")
            st.caption("Cost change over time")
            fc_col1, fc_col2 = st.columns(2)
            with fc_col1:
                vals["fixed_change_pct"] = st.number_input(
                    "Fixed cost change (%)", -50.0, 100.0, 0.0, step=0.5, format="%.1f",
                    key=f"{prefix}_fcp", help="Positive = costs rise over time; negative = decline.")
            with fc_col2:
                vals["fixed_change_mode"] = st.radio(
                    "Compounding", ["Monthly", "Yearly"],
                    key=f"{prefix}_fcm", horizontal=True).lower()

        with st.expander("Variable Costs"):
            vals["trainer_rate"] = st.number_input("Trainer rate ($/class hr)", 15.0, 200.0,
                                                   d.get("trainer_rate", 35.0), step=5.0,
                                                   key=f"{prefix}_trate")
            vals["classes_wk"] = st.number_input("Classes per week", 5, 100, d.get("classes_wk", 30),
                                                 step=1, key=f"{prefix}_cpw")
            vals["trainer_tax"] = st.number_input("Trainer payroll tax (%)", 0.0, 30.0, 12.0,
                                                  step=1.0, format="%.1f", key=f"{prefix}_ttax")
            vals["proc_fee"] = st.number_input("CC processing fee (%)", 0.0, 10.0, 2.9,
                                               step=0.1, format="%.1f", key=f"{prefix}_proc")
            st.markdown("---")
            st.caption("Cost change over time")
            vc_col1, vc_col2 = st.columns(2)
            with vc_col1:
                vals["variable_change_pct"] = st.number_input(
                    "Variable cost change (%)", -50.0, 100.0, 0.0, step=0.5, format="%.1f",
                    key=f"{prefix}_vcp", help="Positive = costs rise over time; negative = decline.")
            with vc_col2:
                vals["variable_change_mode"] = st.radio(
                    "Compounding", ["Monthly", "Yearly"],
                    key=f"{prefix}_vcm", horizontal=True).lower()

    # ── Other Expenses (one-time / irregular) ──
    with st.expander("Other Expenses (one-time)"):
        st.caption("One-time or irregular expenses that occur in specific months.")
        vals["other_amount"] = st.number_input("Amount ($)", 0.0, 500_000.0, 0.0,
                                               step=500.0, format="%.0f", key=f"{prefix}_oea")
        vals["other_months"] = st.multiselect("Months to apply",
                                              options=list(range(1, 25)),
                                              default=[], key=f"{prefix}_oem",
                                              help="Select which months this expense occurs.")

    with st.expander("Tax Settings"):
        vals["ignore_tax"] = st.toggle("Ignore taxes", value=True, key=f"{prefix}_notax")
        vals["tax_rt"] = 25.0
        if not vals["ignore_tax"]:
            vals["tax_rt"] = st.number_input("Effective tax rate (%)", 0.0, 50.0, 25.0,
                                             step=1.0, format="%.1f", key=f"{prefix}_taxrt")

    return vals


def _build_studio_inputs(vals: dict, rent_rate: float) -> StudioInputs:
    """Build a StudioInputs from a unified values dict (loc + settings merged)."""
    v = vals
    fc = FixedCosts(
        rent=v["rent"], manager_wages=v["mgr_wages"],
        manager_payroll_taxes=v["mgr_tax"],
        owner_wages=v["owner_wages"], owner_payroll_taxes=v["owner_tax"],
        professional_fees=v["prof_fees"], insurance=v["ins"],
        accountable_plan_reimb=v["acct_reimb"],
        repairs_maintenance=v["repairs"],
        computer_internet=v["comp_int"], telephone=v["phone"],
        gym_music=v["music"], gym_supplies=v["supplies"],
        office_expenses=v["office"],
        client_processing_software=v["sw"],
        small_equipment=v["sm_equip"],
        automobile_expense=v["auto"], security_fee=v["sec"],
        dues_subscriptions=v["dues"], bank_service_charges=v["bank"],
        licenses_permits=v["lic"], business_taxes=v.get("biz_taxes", 0.0),
        marketing=v["mktg"],
    )
    vc = VariableCosts(
        trainer_hourly_rate=v["trainer_rate"], classes_per_week=v["classes_wk"],
        trainer_payroll_tax_pct=v["trainer_tax"], processing_fee_pct=v["proc_fee"],
    )
    return StudioInputs(
        city=v["city"], state=v["state"], location_label=v.get("label", ""),
        sqft=v["sqft"], arpm=v["arpm"], arpm_year2=v.get("arpm_y2"),
        target_members=v["target"],
        add_ons_enabled=v["addons"], add_on_revenue_per_member=v["addon_rev"],
        extra_income_enabled=v.get("extra_income_on", False),
        extra_income_monthly=v.get("extra_income", 0.0),
        starting_members=v["starting"], monthly_net_adds=v["net_adds"],
        fixed_costs=fc, variable_costs=vc,
        simplified_costs=v.get("simplified_costs", False),
        simplified_fixed_total=v.get("simple_fixed", 0.0),
        simplified_variable_total=v.get("simple_variable", 0.0),
        capex=v.get("capex", 0.0), opening_costs=v.get("opening", 0.0),
        studio_mode=v.get("studio_mode", "new"),
        fixed_cost_change_pct=v.get("fixed_change_pct", 0.0),
        fixed_cost_change_mode=v.get("fixed_change_mode", "yearly"),
        variable_cost_change_pct=v.get("variable_change_pct", 0.0),
        variable_cost_change_mode=v.get("variable_change_mode", "yearly"),
        other_expense_amount=v.get("other_amount", 0.0),
        other_expense_months=v.get("other_months", []),
        ignore_taxes=v["ignore_tax"], tax_rate=v["tax_rt"],
        scenario=v["scenario"].lower(),
        rent_rate_sqft_annual=rent_rate, cam_pct=CFG["rent"]["cam_pct"],
    )


def _resolve_market(prefix: str, city: str, state: str,
                    live_data: bool = True, radius: float = 5.0,
                    trigger: bool = False) -> Optional[MarketData]:
    """Handle market research fetch / cache for a location."""
    mr_key = f"{prefix}_market"
    market = None
    if trigger:
        market = research_location(city, state,
                                   mode="no_key", live_data=live_data,
                                   radius_miles=int(radius))
        st.session_state[mr_key] = market
    elif mr_key in st.session_state:
        market = st.session_state[mr_key]
    return market


def collect_inputs_single(prefix: str, defaults: dict) -> tuple:
    """Full input form for Single Location mode.
    Returns (StudioInputs, Optional[MarketData]).
    """
    d = defaults
    # Location fields at top
    c1, c2 = st.columns([2, 1])
    with c1:
        city = st.text_input("City", value=d.get("city", ""), key=f"{prefix}_city",
                             placeholder="e.g. Austin")
    with c2:
        state = st.text_input("State", value=d.get("state", ""), key=f"{prefix}_state",
                              max_chars=2, placeholder="TX")
    label = st.text_input("Label (optional)", value=d.get("label", ""), key=f"{prefix}_label")
    live_data = st.toggle("Fetch live Census data", value=True, key=f"{prefix}_live",
                          help="Fetches real demographics via Census Bureau FIPS lookup.")
    radius = st.slider("Competitor search radius (miles)", 0.2, 10.0, 5.0,
                       step=0.2, key=f"{prefix}_radius")
    run_mr = st.button("Run Market Research", key=f"{prefix}_mr",
                       use_container_width=True, type="primary")
    arpm = st.number_input("ARPM Year 1 ($)", 50.0, 800.0, d.get("arpm", 199.0), step=5.0,
                           format="%.2f", key=f"{prefix}_arpm")
    rent = st.number_input("Monthly Rent ($)", 0.0, 100_000.0, d.get("rent", 0.0),
                           step=250.0, format="%.0f", key=f"{prefix}_rent",
                           help="Enter 0 to auto-estimate.")

    # All other settings
    settings = _collect_settings(prefix, defaults)

    # Merge location + settings into one dict
    vals = {**settings, "city": city, "state": state, "label": label,
            "rent": rent, "arpm": arpm}

    # Market research
    market = _resolve_market(prefix, city, state, live_data, radius, trigger=run_mr)
    rent_rate = CFG["rent"]["default_rate_per_sqft_annual"]
    if market and market.estimated_rent_sqft_annual:
        rent_rate = market.estimated_rent_sqft_annual

    inp = _build_studio_inputs(vals, rent_rate)
    return inp, market


def _collect_location_full(prefix: str, defaults: dict, loc_label: str = "A") -> dict:
    """Full input form for one location in Compare mode.
    Returns a flat dict with all fields needed to build StudioInputs.
    """
    d = defaults
    city_placeholder = "e.g. Austin" if loc_label == "A" else "e.g. Denver"
    state_placeholder = "TX" if loc_label == "A" else "CO"
    label_placeholder = "e.g. Studio A" if loc_label == "A" else "e.g. Studio B"

    c1, c2 = st.columns([2, 1])
    with c1:
        city = st.text_input("City", value=d.get("city", ""), key=f"{prefix}_city",
                             placeholder=city_placeholder)
    with c2:
        state = st.text_input("State", value=d.get("state", ""), key=f"{prefix}_state",
                              max_chars=2, placeholder=state_placeholder)
    label = st.text_input("Label", value=d.get("label", ""), key=f"{prefix}_label",
                          placeholder=label_placeholder)
    c3, c4 = st.columns(2)
    with c3:
        arpm = st.number_input("ARPM ($)", 50.0, 800.0, d.get("arpm", 199.0), step=5.0,
                               format="%.2f", key=f"{prefix}_arpm")
    with c4:
        rent = st.number_input("Rent ($/mo)", 0.0, 100_000.0, d.get("rent", 0.0),
                               step=250.0, format="%.0f", key=f"{prefix}_rent",
                               help="0 = auto-estimate")
    live_data = st.toggle("Live Census data", value=True, key=f"{prefix}_live")
    radius = st.slider("Competitor radius (miles)", 0.2, 10.0, 5.0,
                       step=0.2, key=f"{prefix}_radius")

    # All settings for this location
    settings = _collect_settings(prefix, defaults)

    # Merge everything
    vals = {**settings, "city": city, "state": state, "label": label,
            "rent": rent, "arpm": arpm, "live_data": live_data, "radius": radius}
    return vals


# ────────────────────────────────────────────────────────────────────
# SECTION 1: Key Metrics
# ────────────────────────────────────────────────────────────────────

def render_key_metrics(inputs: StudioInputs, result: FeasibilityResult,
                       market: Optional[MarketData], scope: str = "main"):
    """Render key financial KPIs."""
    st.subheader("Key Metrics")
    no_startup = inputs.total_startup == 0

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        be_val = fm(result.break_even_month)
        be_sub = f"{fd(result.break_even_revenue)}/mo needed"
        if result.break_even_month is None:
            be_sub = "Studio never turns profitable"
        elif result.break_even_month > 120:
            be_val = f"~{result.break_even_month} mo"
            be_sub = "Estimated — beyond 10-year forecast"
        st.markdown(mc("&#x1F4C5;", "Break-Even Month", be_val, be_sub),
                    unsafe_allow_html=True)
    with r1c2:
        payback_sub = "No startup costs (existing)" if no_startup else f"on {fd(inputs.total_startup)} invested"
        if result.payback_months is None:
            payback_val = "Never"
            payback_sub = "Studio never covers startup costs"
        elif result.payback_months == 0:
            payback_val = "0 (sunk)"
            payback_sub = "Existing studio — costs already paid"
        elif result.payback_months > 120:
            payback_val = f"~{result.payback_months} mo"
            payback_sub = f"Estimated — {fd(inputs.total_startup)} invested"
        else:
            payback_val = fm(result.payback_months)
        st.markdown(mc("&#x1F504;", "Payback Period", payback_val, payback_sub),
                    unsafe_allow_html=True)
    with r1c3:
        roi_label = "1-Year ROI (vs Opex)" if no_startup else "1-Year ROI"
        st.markdown(mc("&#x1F4C8;", roi_label, fp(result.roi_1y),
                        f"3-Year: {fp(result.roi_3y)}"), unsafe_allow_html=True)

    st.write("")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        margin_val = f"{result.profit_margin_m24:.1f}%" if result.profit_margin_m24 is not None else "N/A"
        margin_color = "&#x1F7E2;" if (result.profit_margin_m24 or 0) >= 15 else (
            "&#x1F7E1;" if (result.profit_margin_m24 or 0) >= 0 else "&#x1F534;")
        st.markdown(mc(margin_color, "Profit Margin (Mo 24)",
                        margin_val, "Monthly Profit / Revenue — aim for 15-25%"), unsafe_allow_html=True)
    with r2c2:
        st.markdown(mc("&#x1F465;", "Break-Even Members", str(result.break_even_members or "N/A"),
                        f"Target: {inputs.target_members}"), unsafe_allow_html=True)
    with r2c3:
        ebitda24 = result.pro_forma[-1].ebitda if result.pro_forma else 0
        st.markdown(mc("&#x1F4B0;", "Month 24 Profit", fd(ebitda24),
                        "Projected monthly profit"), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# SECTION 1b: Side-by-side Key Metrics for comparison mode
# ────────────────────────────────────────────────────────────────────

def render_comparison_key_metrics(r1: FeasibilityResult, r2: FeasibilityResult,
                                  i1: StudioInputs, i2: StudioInputs):
    """Side-by-side KPI cards for both locations."""
    st.subheader("Key Metrics — Side-by-Side")

    col_a, col_b = st.columns(2)

    for col, inp, res, label in [(col_a, i1, r1, i1.display_label),
                                  (col_b, i2, r2, i2.display_label)]:
        with col:
            st.markdown(f"**{label}**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(mc("&#x1F4C5;", "Break-Even", fm(res.break_even_month), ""), unsafe_allow_html=True)
                margin_val = f"{res.profit_margin_m24:.1f}%" if res.profit_margin_m24 is not None else "N/A"
                st.markdown(mc("&#x1F4C8;", "Profit Margin", margin_val, ""), unsafe_allow_html=True)
            with c2:
                st.markdown(mc("&#x1F504;", "Payback", fm(res.payback_months), ""), unsafe_allow_html=True)
                ebitda24 = res.pro_forma[-1].ebitda if res.pro_forma else 0
                st.markdown(mc("&#x1F4B0;", "Mo 24 Profit", fd(ebitda24), ""), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# SECTION 2: Feasibility Score + Explanation
# ────────────────────────────────────────────────────────────────────

def _score_bar_html(label: str, pts: float, max_pts: float, color: str, display: str) -> str:
    bar_pct = int((pts / max_pts) * 100) if max_pts > 0 else 0
    return (
        f'<div class="score-bar">'
        f'<span class="score-bar-label">{label} ({display})</span>'
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{bar_pct}%; background:{color};"></div>'
        f'</div>'
        f'<span class="score-bar-pts">{pts:.0f} pts</span>'
        f'</div>')


def _score_component(label, value, green_thresh, yellow_thresh, max_pts,
                     higher_is_better=True, suffix=""):
    if value is None:
        return (0.0, "#e2e8f0", "N/A")
    if higher_is_better:
        if value >= green_thresh:
            pts, color = max_pts, "#22c55e"
        elif value >= yellow_thresh:
            pts, color = max_pts * 0.6, "#eab308"
        else:
            pts, color = max_pts * 0.15, "#ef4444"
    else:
        if value <= green_thresh:
            pts, color = max_pts, "#22c55e"
        elif value <= yellow_thresh:
            pts, color = max_pts * 0.6, "#eab308"
        else:
            pts, color = max_pts * 0.15, "#ef4444"
    return (pts, color, f"{value}{suffix}")


def render_feasibility_score(result: FeasibilityResult, thresholds: dict,
                             inputs: Optional[StudioInputs] = None,
                             market: Optional[MarketData] = None,
                             scope: str = "main"):
    st.subheader("Feasibility Score")

    st.markdown(
        f'<div style="text-align:center; margin:0.5rem 0 1rem 0;">'
        f'{tl_badge(result.traffic_light, result.feasibility_score)}</div>',
        unsafe_allow_html=True)

    components = [
        ("Operating Break-Even", result.ebitda_positive_month,
         thresholds["ebitda_month_green"], thresholds["ebitda_month_yellow"], 25.0, False, " mo",),
        ("Profit Margin (Mo 24)", result.profit_margin_m24,
         thresholds["margin_green"], thresholds["margin_yellow"], 25.0, True, "%"),
        ("Payback Period", result.payback_months,
         thresholds["payback_green"], thresholds["payback_yellow"], 25.0, False, " mo"),
        ("1-Year ROI", result.roi_1y,
         thresholds["roi_1y_green"], thresholds["roi_1y_yellow"], 25.0, True, "%"),
    ]

    bars_html = ""
    for label, value, g_t, y_t, max_pts, higher_better, suffix in components:
        pts, color, display = _score_component(label, value, g_t, y_t, max_pts, higher_better, suffix)
        bars_html += _score_bar_html(label, pts, max_pts, color, display)

    st.markdown(
        f'<div class="score-explain">'
        f'<h4>How the Score Works</h4>'
        f'<p style="font-size:0.82rem; color:#64748b; margin-bottom:12px;">'
        f'The score (0-100) is the sum of four equally weighted components. '
        f'Each earns up to 25 points based on how it compares to industry benchmarks.</p>'
        f'{bars_html}'
        f'<div style="margin-top:12px; padding-top:10px; border-top:1px solid #e2e8f0; font-size:0.8rem; color:#64748b;">'
        f'<strong style="color:#166534;">GO (70+):</strong> Strong financials, proceed with confidence &nbsp;|&nbsp; '
        f'<strong style="color:#854d0e;">CAUTION (40-69):</strong> Viable but review assumptions &nbsp;|&nbsp; '
        f'<strong style="color:#991b1b;">NO-GO (&lt;40):</strong> High risk, reconsider or renegotiate'
        f'</div></div>',
        unsafe_allow_html=True)

    with st.expander("View scoring thresholds", expanded=False):
        st.markdown(
            f"**Operating Break-Even:** Full points \u2264 month {thresholds['ebitda_month_green']}, "
            f"partial \u2264 month {thresholds['ebitda_month_yellow']}, minimal otherwise.\n\n"
            f"**Profit Margin (Mo 24):** Full points \u2265 {thresholds['margin_green']}%, "
            f"partial \u2265 {thresholds['margin_yellow']}%, minimal otherwise.\n\n"
            f"**Payback Period:** Full points \u2264 {thresholds['payback_green']} mo, "
            f"partial \u2264 {thresholds['payback_yellow']} mo, minimal otherwise.\n\n"
            f"**1-Year ROI:** Full points \u2265 {thresholds['roi_1y_green']}%, "
            f"partial \u2265 {thresholds['roi_1y_yellow']}%, minimal otherwise.")


# ────────────────────────────────────────────────────────────────────
# SECTION 3: Market Research
# ────────────────────────────────────────────────────────────────────

def render_market_research(market: Optional[MarketData], inputs: StudioInputs,
                           scope: str = "main"):
    st.subheader("Market Research")

    if not market:
        st.info("Click **Run Market Research** in the sidebar to pull demographics for this location.")
        return

    st.markdown(source_badge(market), unsafe_allow_html=True)
    st.caption(market.data_source_detail)
    st.write("")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        pop_str = f"{market.population:,}" if market.population else "N/A"
        st.markdown(mc("&#x1F3D8;", "Population", pop_str, ""), unsafe_allow_html=True)
    with c2:
        st.markdown(mc("&#x1F4B0;", "Median HH Income", fd(market.median_household_income), "Annual"), unsafe_allow_html=True)
    with c3:
        dens = f"{market.population_density:,.0f}/mi\u00B2" if market.population_density else "N/A"
        st.markdown(mc("&#x1F4CD;", "Pop. Density", dens, ""), unsafe_allow_html=True)
    with c4:
        age_str = f"{market.median_age:.1f} yrs" if market.median_age else "N/A"
        st.markdown(mc("&#x1F382;", "Median Age", age_str, ""), unsafe_allow_html=True)
    with c5:
        pct_str = f"{market.pct_age_25_54:.1f}%" if market.pct_age_25_54 else "N/A"
        st.markdown(mc("&#x1F3CB;", "Ages 25-54", pct_str, "Fitness target demo"), unsafe_allow_html=True)

    st.write("")

    c6, c7, c8, c9 = st.columns(4)
    with c6:
        st.markdown(mc("&#x1F3E2;", "Est. Rent $/sqft/yr", fd(market.estimated_rent_sqft_annual), market.rent_source), unsafe_allow_html=True)
    with c7:
        st.markdown(mc("&#x1F94A;", "Competitors Nearby", str(market.competition_count), market.competition_source), unsafe_allow_html=True)
    with c8:
        arpm_range = f"${market.suggested_arpm_low:.0f}-${market.suggested_arpm_high:.0f}" if market.suggested_arpm_low else "N/A"
        st.markdown(mc("&#x1F4B2;", "Suggested ARPM", arpm_range, "Based on local income"), unsafe_allow_html=True)
    with c9:
        d_icon = {"High": "&#x1F7E2;", "Medium": "&#x1F7E1;", "Low": "&#x1F534;"}.get(market.demand_potential, "&#x26AA;")
        st.markdown(mc(d_icon, "Demand Potential", market.demand_potential, ""), unsafe_allow_html=True)

    if market.competition_details:
        with st.expander(f"View {market.competition_count} Competitors Nearby", expanded=False):
            st.caption(market.competition_source)
            for comp in market.competition_details:
                name = comp.get("name", comp.get("keyword", "Unknown"))
                cat = comp.get("category", "")
                dist = comp.get("distance_miles")
                count = comp.get("estimated_count")
                if dist is not None:
                    st.markdown(
                        f'<div style="padding:4px 0; border-bottom:1px solid #f1f5f9;">'
                        f'<strong>{name}</strong>'
                        f'<span style="color:#64748b; font-size:0.85rem;"> \u2014 {cat}</span>'
                        f'<span style="float:right; color:#94a3b8; font-size:0.82rem;">{dist} mi</span>'
                        f'</div>',
                        unsafe_allow_html=True)
                elif count is not None:
                    st.markdown(
                        f'<div style="padding:4px 0; border-bottom:1px solid #f1f5f9;">'
                        f'<strong>{name}</strong>'
                        f'<span style="float:right; color:#94a3b8; font-size:0.82rem;">~{count} estimated</span>'
                        f'</div>',
                        unsafe_allow_html=True)

    if market.neighborhood_suggestions:
        with st.expander("Best Neighborhood Options", expanded=False):
            for nb in market.neighborhood_suggestions:
                score_color = "#22c55e" if nb["score"] >= 70 else "#eab308" if nb["score"] >= 50 else "#ef4444"
                st.markdown(
                    f'<div class="nbhood-card">'
                    f'<span class="nbhood-score" style="color:{score_color};">{nb["score"]:.0f}/100</span> '
                    f'<strong>{nb["area"]}</strong><br>'
                    f'<span style="font-size:0.82rem; color:#64748b;">{nb["rationale"]}</span><br>'
                    f'<span style="font-size:0.75rem; color:#94a3b8;">'
                    f'Income match: {nb["income_match"]} &nbsp;|&nbsp; Density match: {nb["density_match"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True)

    for w in (market.warnings or []):
        st.warning(w)


# ────────────────────────────────────────────────────────────────────
# SECTION 4: Expense Tracker
# ────────────────────────────────────────────────────────────────────

def render_expense_tracker(inputs: StudioInputs, result: FeasibilityResult,
                           scope: str = "main"):
    st.subheader("Expense Tracker")

    last_row = result.pro_forma[-1] if result.pro_forma else None

    if inputs.simplified_costs:
        st.caption("Using simplified cost totals.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<table class="cost-table">'
                f'<tr><th colspan="2">Fixed Costs</th><th style="text-align:right">Monthly</th></tr>'
                f'{_cost_row("&#x1F4CB;", "Total Fixed (user-entered)", inputs.simplified_fixed_total)}'
                f'{_cost_total_row("Total Fixed", inputs.simplified_fixed_total)}</table>',
                unsafe_allow_html=True)
        with col2:
            st.markdown(
                f'<table class="cost-table">'
                f'<tr><th colspan="2">Variable Costs</th><th style="text-align:right">Monthly</th></tr>'
                f'{_cost_row("&#x1F4CB;", "Total Variable (user-entered)", inputs.simplified_variable_total)}'
                f'{_cost_total_row("Total Variable", inputs.simplified_variable_total)}</table>',
                unsafe_allow_html=True)
        grand_total = inputs.simplified_fixed_total + inputs.simplified_variable_total
    else:
        st.caption("Monthly operating costs organized by type.")
        fc = inputs.fixed_costs
        vc = inputs.variable_costs

        col1, col2 = st.columns(2)

        with col1:
            fixed_items = [
                ("&#x1F3E0;", "Rent & Lease", fc.rent),
                ("&#x1F464;", "Manager Wages", fc.manager_wages),
                ("&#x1F4C4;", "Manager Payroll Tax", fc.manager_payroll_taxes),
                ("&#x1F468;&#x200D;&#x1F4BC;", "Owner Wages", fc.owner_wages),
                ("&#x1F4C4;", "Owner Payroll Tax", fc.owner_payroll_taxes),
                ("&#x2696;", "Professional Fees", fc.professional_fees),
                ("&#x1F6E1;", "Insurance", fc.insurance),
                ("&#x1F4CB;", "Accountable Plan", fc.accountable_plan_reimb),
                ("&#x1F4BB;", "Computer & Internet", fc.computer_internet),
                ("&#x260E;", "Telephone", fc.telephone),
                ("&#x1F3B5;", "Music License", fc.gym_music),
                ("&#x1F4DD;", "Office Expenses", fc.office_expenses),
                ("&#x2699;", "Processing Software", fc.client_processing_software),
                ("&#x1F697;", "Automobile", fc.automobile_expense),
                ("&#x1F512;", "Security Fee", fc.security_fee),
                ("&#x1F4DA;", "Dues & Subs", fc.dues_subscriptions),
                ("&#x1F3E6;", "Bank Charges", fc.bank_service_charges),
                ("&#x1F4DC;", "Licenses & Permits", fc.licenses_permits),
                ("&#x1F4B8;", "Business Taxes", fc.business_taxes),
                ("&#x1F4E3;", "Marketing", fc.marketing),
                ("&#x1F527;", "Repairs & Maint.", fc.repairs_maintenance),
                ("&#x1F9F9;", "Gym Supplies", fc.gym_supplies),
                ("&#x1F3CB;", "Small Equipment", fc.small_equipment),
            ]
            fixed_total = sum(v for _, _, v in fixed_items)
            rows_html = "".join(_cost_row(i, n, v) for i, n, v in fixed_items if v > 0)
            st.markdown(
                f'<table class="cost-table">'
                f'<tr><th colspan="2">Fixed Costs</th><th style="text-align:right">Monthly</th></tr>'
                f'{rows_html}{_cost_total_row("Total Fixed", fixed_total)}</table>',
                unsafe_allow_html=True)

        with col2:
            trainer_cost = vc.monthly_trainer_cost()
            current_rev = last_row.revenue_total if last_row else 0
            proc_fee_val = vc.monthly_processing_fee(current_rev)

            var_items = [
                ("&#x1F3C3;", "Trainer Wages + Tax", trainer_cost),
                ("&#x1F4B3;", "CC Processing Fees", proc_fee_val),
            ]
            var_total = sum(v for _, _, v in var_items)
            rows_html = "".join(_cost_row(i, n, v) for i, n, v in var_items if v > 0)
            st.markdown(
                f'<table class="cost-table">'
                f'<tr><th colspan="2">Variable Costs</th><th style="text-align:right">Monthly</th></tr>'
                f'{rows_html}{_cost_total_row("Total Variable", var_total)}</table>',
                unsafe_allow_html=True)
            st.caption("Trainer = rate x classes/wk x 4.33. CC fees = % of revenue.")

        grand_total = fixed_total + var_total

    # ── Cost change info ──
    has_fixed_change = inputs.fixed_cost_change_pct != 0
    has_var_change = inputs.variable_cost_change_pct != 0
    has_other = inputs.other_expense_amount > 0 and len(inputs.other_expense_months) > 0

    if has_fixed_change or has_var_change:
        st.write("")
        st.caption("Cost changes over time:")
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            if has_fixed_change:
                sign = "+" if inputs.fixed_cost_change_pct > 0 else ""
                st.markdown(
                    f'<div style="background:#fef9c3; border:1px solid #fde68a; border-radius:8px; '
                    f'padding:0.6rem 0.8rem; font-size:0.85rem;">'
                    f'<strong>Fixed Costs:</strong> {sign}{inputs.fixed_cost_change_pct:.1f}% '
                    f'{inputs.fixed_cost_change_mode} compounding<br>'
                    f'<span style="color:#64748b;">Mo 1: ${last_row.fixed_costs_total if last_row else 0:,.0f} '
                    f'&rarr; Mo 24 (with change): ${result.pro_forma[-1].fixed_costs_total if result.pro_forma else 0:,.0f}</span>'
                    f'</div>', unsafe_allow_html=True)
        with ch_col2:
            if has_var_change:
                sign = "+" if inputs.variable_cost_change_pct > 0 else ""
                st.markdown(
                    f'<div style="background:#fef9c3; border:1px solid #fde68a; border-radius:8px; '
                    f'padding:0.6rem 0.8rem; font-size:0.85rem;">'
                    f'<strong>Variable Costs:</strong> {sign}{inputs.variable_cost_change_pct:.1f}% '
                    f'{inputs.variable_cost_change_mode} compounding<br>'
                    f'<span style="color:#64748b;">Mo 1: ${result.pro_forma[0].variable_costs_total if result.pro_forma else 0:,.0f} '
                    f'&rarr; Mo 24: ${result.pro_forma[-1].variable_costs_total if result.pro_forma else 0:,.0f}</span>'
                    f'</div>', unsafe_allow_html=True)

    if has_other:
        st.write("")
        st.caption("One-time / irregular expenses:")
        months_str = ", ".join(str(m) for m in sorted(inputs.other_expense_months))
        st.markdown(
            f'<table class="cost-table">'
            f'<tr><th colspan="2">Other Expenses</th><th style="text-align:right">Amount</th></tr>'
            f'{_cost_row("&#x1F4CB;", f"Months: {months_str}", inputs.other_expense_amount)}'
            f'{_cost_total_row("Per occurrence", inputs.other_expense_amount)}</table>',
            unsafe_allow_html=True)

    st.markdown(
        f'<div style="background:#f1f5f9; border-radius:8px; padding:0.75rem 1rem; '
        f'margin-top:0.75rem; display:flex; justify-content:space-between; align-items:center;">'
        f'<span style="font-weight:700; color:#334155;">Total Monthly Operating Costs (Mo 24)</span>'
        f'<span style="font-weight:700; color:#0f172a; font-size:1.2rem;">'
        f'${last_row.total_opex:,.0f}</span></div>' if last_row else '',
        unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# SECTION 5: 24-Month Projections
# ────────────────────────────────────────────────────────────────────

def render_charts(inputs: StudioInputs, result: FeasibilityResult,
                  scope: str = "main"):
    df = pd.DataFrame([{
        "Month": r.month, "Members": r.members,
        "Revenue": r.revenue_total, "Total OpEx": r.total_opex,
        "Monthly Profit": r.ebitda, "Net Monthly Income": r.operating_cf,
        "Total Profit to Date": r.cumulative_cf,
    } for r in result.pro_forma])

    st.subheader("24-Month Projections")
    taxes_off = inputs.ignore_taxes
    if taxes_off:
        st.caption("Taxes are off — Net Monthly Income equals Monthly Profit. "
                   "Turn on taxes in the sidebar to see after-tax figures.")

    # Find the first month where Revenue >= Total OpEx (operating break-even)
    be_month = None
    be_revenue = None
    for r in result.pro_forma:
        if r.revenue_total >= r.total_opex:
            be_month = r.month
            be_revenue = r.revenue_total
            break

    cc1, cc2 = st.columns(2)
    with cc1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Month"], y=df["Revenue"], name="Revenue", marker_color="#3b82f6"))
        fig.add_trace(go.Bar(x=df["Month"], y=df["Total OpEx"], name="OpEx", marker_color="#f87171"))

        # Highlight break-even month with vertical line and annotation
        if be_month is not None:
            fig.add_vline(x=be_month, line_dash="dash", line_color="#16a34a", line_width=2)
            fig.add_annotation(
                x=be_month, y=1.12, xref="x", yref="paper",
                text=f"<b>Break-Even: Month {be_month}</b>",
                showarrow=False, font=dict(size=13, color="#16a34a"),
                bgcolor="rgba(255,255,255,0.85)", bordercolor="#16a34a",
                borderwidth=1, borderpad=4,
            )

        title_text = "Revenue vs Expenses"
        fig.update_layout(title=dict(text=title_text, font=dict(size=14)),
                          barmode="group", template="plotly_white", height=340,
                          margin=dict(t=65, b=30, l=50, r=20),
                          legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
                          yaxis_tickprefix="$", yaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True, key=f"{scope}_revenue_vs_opex")

    with cc2:
        fig2 = go.Figure()
        clrs = ["#22c55e" if v >= 0 else "#ef4444" for v in df["Net Monthly Income"]]
        fig2.add_trace(go.Bar(x=df["Month"], y=df["Net Monthly Income"], name="Net Monthly Income", marker_color=clrs))
        fig2.add_trace(go.Scatter(x=df["Month"], y=df["Total Profit to Date"], name="Total Profit to Date",
                                  line=dict(color="#7c3aed", width=3), mode="lines"))
        fig2.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        fig2.update_layout(title=dict(text="Income & Total Profit", font=dict(size=14)),
                           template="plotly_white", height=320,
                           margin=dict(t=50, b=30, l=50, r=20),
                           legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
                           yaxis_tickprefix="$", yaxis_tickformat=",")
        st.plotly_chart(fig2, use_container_width=True, key=f"{scope}_cashflow")

    cc3, cc4 = st.columns(2)
    with cc3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df["Month"], y=df["Members"], fill="tozeroy",
                                  line=dict(color="#0ea5e9", width=2),
                                  fillcolor="rgba(14,165,233,0.15)", name="Members"))
        fig3.add_hline(y=inputs.target_members, line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"Target: {inputs.target_members}")
        fig3.update_layout(title=dict(text="Member Growth", font=dict(size=14)),
                           template="plotly_white", height=320,
                           margin=dict(t=50, b=30, l=50, r=20))
        st.plotly_chart(fig3, use_container_width=True, key=f"{scope}_member_growth")

    with cc4:
        clrs2 = ["#22c55e" if v >= 0 else "#ef4444" for v in df["Monthly Profit"]]
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=df["Month"], y=df["Monthly Profit"], marker_color=clrs2, name="Monthly Profit"))
        fig4.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        fig4.update_layout(title=dict(text="Monthly Profit", font=dict(size=14)),
                           template="plotly_white", height=320,
                           margin=dict(t=50, b=30, l=50, r=20),
                           yaxis_tickprefix="$", yaxis_tickformat=",")
        st.plotly_chart(fig4, use_container_width=True, key=f"{scope}_ebitda")


# ────────────────────────────────────────────────────────────────────
# SECTION 6: Sensitivity Analysis
# ────────────────────────────────────────────────────────────────────

def render_sensitivity(inputs: StudioInputs, scope: str = "main"):
    st.subheader("Sensitivity Analysis")
    st.caption("How changing one input at a time affects your 1-Year ROI.")
    tabs = st.tabs(["ARPM", "Rent", "Marketing", "Trainer Rate", "Net Adds"])
    cfgs = [
        ("arpm", "ARPM ($)",
         [v for v in range(int(inputs.arpm * 0.6), int(inputs.arpm * 1.4) + 1, int(max(1, inputs.arpm * 0.1)))]),
        ("rent", "Rent ($)",
         [round(inputs.fixed_costs.rent * f) for f in [0.7, 0.85, 1.0, 1.15, 1.3]] if inputs.fixed_costs.rent > 0 else []),
        ("marketing", "Marketing ($)",
         [round(inputs.fixed_costs.marketing * f) for f in [0.5, 0.75, 1.0, 1.25, 1.5]]),
        ("trainer_hourly_rate", "Trainer $/hr",
         [round(inputs.variable_costs.trainer_hourly_rate * f) for f in [0.7, 0.85, 1.0, 1.15, 1.3]]),
        ("monthly_net_adds", "Net Adds/mo",
         list(range(max(1, inputs.monthly_net_adds - 6), inputs.monthly_net_adds + 7, 2))),
    ]
    for idx, (tab, (param, label, vals)) in enumerate(zip(tabs, cfgs)):
        with tab:
            if vals:
                res = sensitivity(inputs, param, vals)
                sdf = pd.DataFrame(res)
                sdf.rename(columns={"value": label}, inplace=True)
                figS = go.Figure()
                figS.add_trace(go.Scatter(x=sdf[label], y=sdf["roi_1y"], mode="lines+markers",
                                          line=dict(color="#8b5cf6", width=2.5), marker=dict(size=7)))
                figS.update_layout(template="plotly_white", height=260,
                                   yaxis_title="1Y ROI %", xaxis_title=label,
                                   margin=dict(t=20, b=40, l=50, r=20))
                st.plotly_chart(figS, use_container_width=True,
                                key=f"{scope}_sens_{param}_{idx}")
                disp = sdf[[c for c in [label, "ebitda_m24", "cum_cf_m24", "break_even_month",
                                        "payback_months", "roi_1y"] if c in sdf.columns]].copy()
                disp.columns = [label, "Profit (M24)", "Total Profit (M24)", "BE Month", "Payback", "1Y ROI %"]
                st.dataframe(disp.style.format({"Profit (M24)": "${:,.0f}",
                                                "Total Profit (M24)": "${:,.0f}",
                                                "1Y ROI %": "{:.1f}%"}),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Set a rent value to enable rent sensitivity.")


# ────────────────────────────────────────────────────────────────────
# SECTION 7: Pro Forma + Exports
# ────────────────────────────────────────────────────────────────────

def render_proforma_and_exports(inputs: StudioInputs, result: FeasibilityResult,
                                market: Optional[MarketData], scope: str = "main"):
    st.subheader("Full Pro Forma Table")
    if inputs.ignore_taxes:
        st.caption("Taxes are off — Net Income = Monthly Profit. Enable taxes in the sidebar for after-tax view.")
    has_extra = any(r.revenue_extra > 0 for r in result.pro_forma)
    has_other = any(r.other_expense > 0 for r in result.pro_forma)
    has_taxes = not inputs.ignore_taxes and any(r.taxes > 0 for r in result.pro_forma)
    full_df = pd.DataFrame([{
        "Mo": r.month, "Members": r.members, "ARPM": r.arpm_used,
        **({"Extra Income": r.revenue_extra} if has_extra else {}),
        "Revenue": r.revenue_total, "Fixed": r.fixed_costs_total,
        "Variable": r.variable_costs_total,
        **({"Other": r.other_expense} if has_other else {}),
        "Total OpEx": r.total_opex,
        "Monthly Profit": r.ebitda,
        **({"Taxes": r.taxes} if has_taxes else {}),
        "Net Income": r.operating_cf, "Total Profit": r.cumulative_cf,
    } for r in result.pro_forma])
    money = [c for c in full_df.columns if c not in ("Mo", "Members")]
    fmt = {c: "${:,.0f}" for c in money}
    fmt["Members"] = "{:,.0f}"
    fmt["ARPM"] = "${:,.2f}"
    st.dataframe(full_df.style.format(fmt), use_container_width=True, hide_index=True, height=400)

    st.subheader("Export")
    e1, e2 = st.columns(2)
    with e1:
        buf = io.StringIO()
        full_df.to_csv(buf, index=False)
        st.download_button("Download CSV", buf.getvalue(),
                           file_name=f"proforma_{inputs.city}_{scope}.csv",
                           mime="text/csv", use_container_width=True,
                           key=f"{scope}_dl_csv")
    with e2:
        exp = {
            "location": inputs.display_label,
            "results": {
                "be_members": result.break_even_members, "be_month": result.break_even_month,
                "payback": result.payback_months, "roi_1y": result.roi_1y,
                "roi_3y": result.roi_3y,
                "profit_margin_m24": result.profit_margin_m24,
                "monthly_profit_m24": result.monthly_profit_m24,
                "score": result.feasibility_score, "light": result.traffic_light,
            },
            "market_research": market.to_dict() if market else None,
        }
        st.download_button("Download JSON", json.dumps(exp, indent=2, default=str),
                           file_name=f"feasibility_{inputs.city}_{scope}.json",
                           mime="application/json", use_container_width=True,
                           key=f"{scope}_dl_json")


# ────────────────────────────────────────────────────────────────────
# Full dashboard — enforces section order, passes scope everywhere
# ────────────────────────────────────────────────────────────────────

def render_dashboard(inputs: StudioInputs, result: FeasibilityResult,
                     market: Optional[MarketData], thresholds: dict,
                     scope: str = "main"):
    st.markdown(
        f'<div style="text-align:center; padding:0.25rem 0;">'
        f'<h3 style="margin:0; color:#0f172a;">{inputs.display_label}</h3>'
        f'<p style="color:#64748b; font-size:0.85rem; margin:2px 0 0 0;">'
        f'{inputs.sqft:,} sq ft &nbsp;|&nbsp; ARPM ${inputs.arpm:,.2f}'
        f'{" &rarr; $" + f"{inputs.arpm_year2:,.2f}" if inputs.arpm_year2 else ""}'
        f' &nbsp;|&nbsp; {inputs.scenario.title()}'
        f'{" &nbsp;|&nbsp; Existing Studio" if inputs.studio_mode == "existing" else ""}'
        f'</p></div>',
        unsafe_allow_html=True)
    st.write("")

    render_key_metrics(inputs, result, market, scope=scope)
    st.divider()
    render_feasibility_score(result, thresholds, inputs=inputs, market=market, scope=scope)
    st.divider()
    render_market_research(market, inputs, scope=scope)
    st.divider()
    render_expense_tracker(inputs, result, scope=scope)
    st.divider()
    render_charts(inputs, result, scope=scope)
    st.divider()
    render_sensitivity(inputs, scope=scope)
    st.divider()
    render_proforma_and_exports(inputs, result, market, scope=scope)


# ────────────────────────────────────────────────────────────────────
# Comparison renderer
# ────────────────────────────────────────────────────────────────────

def render_comparison(r1: FeasibilityResult, r2: FeasibilityResult,
                      i1: StudioInputs, i2: StudioInputs,
                      m1: Optional[MarketData], m2: Optional[MarketData]):
    col1, col2 = st.columns(2)
    with col1:
        cls1 = "cmp-winner" if r1.feasibility_score >= r2.feasibility_score else "cmp-loser"
        st.markdown(f'<div class="{cls1}" style="text-align:center; padding:1rem;">'
                    f'<strong>{i1.display_label}</strong><br>'
                    f'{tl_badge(r1.traffic_light, r1.feasibility_score)}</div>',
                    unsafe_allow_html=True)
    with col2:
        cls2 = "cmp-winner" if r2.feasibility_score >= r1.feasibility_score else "cmp-loser"
        st.markdown(f'<div class="{cls2}" style="text-align:center; padding:1rem;">'
                    f'<strong>{i2.display_label}</strong><br>'
                    f'{tl_badge(r2.traffic_light, r2.feasibility_score)}</div>',
                    unsafe_allow_html=True)
    st.write("")

    st.markdown("**Financial Comparison**")
    fin_metrics = [
        ("Feasibility Score", r1.feasibility_score, r2.feasibility_score, "{:.0f}/100", True),
        ("Break-Even Month", r1.break_even_month, r2.break_even_month, "{} mo", False),
        ("Payback Period", r1.payback_months, r2.payback_months, "{} mo", False),
        ("1-Year ROI", r1.roi_1y, r2.roi_1y, "{:.1f}%", True),
        ("3-Year ROI", r1.roi_3y, r2.roi_3y, "{:.1f}%", True),
        ("Profit Margin (Mo 24)", r1.profit_margin_m24, r2.profit_margin_m24, "{:.1f}%", True),
        ("BE Members", r1.break_even_members, r2.break_even_members, "{}", False),
        ("Month 24 Profit", r1.pro_forma[-1].ebitda if r1.pro_forma else 0,
                            r2.pro_forma[-1].ebitda if r2.pro_forma else 0, "${:,.0f}", True),
        ("Monthly Rent", i1.fixed_costs.rent, i2.fixed_costs.rent, "${:,.0f}", False),
        ("Total Startup", i1.total_startup, i2.total_startup, "${:,.0f}", False),
    ]

    rows = []
    for name, v1, v2, fmt, higher_better in fin_metrics:
        s1 = fmt.format(v1) if v1 is not None else "N/A"
        s2 = fmt.format(v2) if v2 is not None else "N/A"
        winner = ""
        if v1 is not None and v2 is not None:
            if higher_better:
                winner = i1.display_label if v1 > v2 else (i2.display_label if v2 > v1 else "Tie")
            else:
                winner = i1.display_label if v1 < v2 else (i2.display_label if v2 < v1 else "Tie")
        rows.append({"Metric": name, i1.display_label: s1, i2.display_label: s2, "Better": winner})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if m1 and m2:
        st.markdown("**Demographics Comparison**")
        demo_rows = [
            ("Population", m1.population, m2.population, "{:,}", True),
            ("Median HH Income", m1.median_household_income, m2.median_household_income, "${:,.0f}", True),
            ("Pop. Density", m1.population_density, m2.population_density, "{:,.0f}/mi\u00B2", True),
            ("Median Age", m1.median_age, m2.median_age, "{:.1f} yrs", False),
            ("Ages 25-54 %", m1.pct_age_25_54, m2.pct_age_25_54, "{:.1f}%", True),
            ("Est. Rent $/sqft/yr", m1.estimated_rent_sqft_annual, m2.estimated_rent_sqft_annual, "${:.0f}", False),
            ("Competitors Nearby", m1.competition_count, m2.competition_count, "{}", False),
            ("Demand Potential", m1.demand_potential, m2.demand_potential, "{}", None),
        ]
        demo_table = []
        for name, v1, v2, fmt, higher_better in demo_rows:
            s1 = fmt.format(v1) if v1 is not None else "N/A"
            s2 = fmt.format(v2) if v2 is not None else "N/A"
            winner = ""
            if higher_better is not None and v1 is not None and v2 is not None:
                try:
                    if higher_better:
                        winner = i1.display_label if float(v1) > float(v2) else (i2.display_label if float(v2) > float(v1) else "Tie")
                    else:
                        winner = i1.display_label if float(v1) < float(v2) else (i2.display_label if float(v2) < float(v1) else "Tie")
                except (ValueError, TypeError):
                    pass
            demo_table.append({"Metric": name, i1.display_label: s1, i2.display_label: s2, "Better": winner})
        st.dataframe(pd.DataFrame(demo_table), use_container_width=True, hide_index=True)

    st.write("")

    st.markdown("**Total Profit to Date — Comparison**")
    fig = go.Figure()
    for res, inp, color in [(r1, i1, "#3b82f6"), (r2, i2, "#f59e0b")]:
        fig.add_trace(go.Scatter(x=[r.month for r in res.pro_forma],
                                 y=[r.cumulative_cf for r in res.pro_forma],
                                 name=inp.display_label,
                                 line=dict(color=color, width=3), mode="lines"))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(template="plotly_white", height=350,
                      margin=dict(t=20, b=30, l=50, r=20),
                      yaxis_tickprefix="$", yaxis_tickformat=",",
                      xaxis_title="Month", yaxis_title="Total Profit to Date ($)")
    st.plotly_chart(fig, use_container_width=True, key="cmp_cumcf")

    st.markdown("**Member Growth Comparison**")
    fig2 = go.Figure()
    for res, inp, color in [(r1, i1, "#3b82f6"), (r2, i2, "#f59e0b")]:
        fig2.add_trace(go.Scatter(x=[r.month for r in res.pro_forma],
                                  y=[r.members for r in res.pro_forma],
                                  name=inp.display_label,
                                  line=dict(color=color, width=3), mode="lines"))
    fig2.update_layout(template="plotly_white", height=300,
                       margin=dict(t=20, b=30, l=50, r=20),
                       xaxis_title="Month", yaxis_title="Members")
    st.plotly_chart(fig2, use_container_width=True, key="cmp_members")


# ────────────────────────────────────────────────────────────────────
# MAIN APP
# ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Studio Feasibility")
    st.caption("Enter studio details below. Works for any U.S. city.")
    st.divider()
    mode = st.radio("Mode", ["Single Location", "Compare Two Locations"],
                    key="app_mode", horizontal=True)
    st.divider()

thresholds = {
    "ebitda_month_green": 10, "ebitda_month_yellow": 18,
    "margin_green": 15.0, "margin_yellow": 0.0,
    "payback_green": 24, "payback_yellow": 36,
    "roi_1y_green": 0.0, "roi_1y_yellow": -20.0,
}

_DEFAULTS = {
    "city": "", "state": "", "label": "",
    "arpm": 199.0, "rent": 0.0,
}

if mode == "Single Location":
    with st.sidebar:
        inputs_a, market_a = collect_inputs_single("s", _DEFAULTS)

    if not inputs_a.city or not inputs_a.state:
        st.info("Enter a **City** and **State** in the sidebar to get started.")
        st.stop()

    result_a = run_model(inputs_a, months=24, thresholds=thresholds)
    render_dashboard(inputs_a, result_a, market_a, thresholds, scope="single")

else:
    # ── Compare mode: each location has its OWN full set of inputs ──
    with st.sidebar:
        st.markdown(
            '<div style="background:#079DD9; color:white; padding:6px 14px; '
            'border-radius:8px; font-weight:700; font-size:0.9rem; margin-bottom:4px;">'
            '\U0001f4cd Location A</div>', unsafe_allow_html=True)
        vals_a = _collect_location_full("ca", _DEFAULTS, loc_label="A")

        st.markdown("---")

        st.markdown(
            '<div style="background:#f59e0b; color:white; padding:6px 14px; '
            'border-radius:8px; font-weight:700; font-size:0.9rem; margin-bottom:4px;">'
            '\U0001f4cd Location B</div>', unsafe_allow_html=True)
        vals_b = _collect_location_full("cb", _DEFAULTS, loc_label="B")

    # ── Compare button in main area ──
    compare_clicked = st.button("Research & Compare", type="primary",
                                use_container_width=True, key="cmp_btn")

    # Run market research when Compare button is clicked
    market_a = _resolve_market("ca", vals_a.get("city", ""), vals_a.get("state", ""),
                               vals_a.get("live_data", True), vals_a.get("radius", 5.0),
                               trigger=compare_clicked)
    market_b = _resolve_market("cb", vals_b.get("city", ""), vals_b.get("state", ""),
                               vals_b.get("live_data", True), vals_b.get("radius", 5.0),
                               trigger=compare_clicked)

    # Build StudioInputs
    rent_rate = CFG["rent"]["default_rate_per_sqft_annual"]
    rent_rate_a = market_a.estimated_rent_sqft_annual if market_a and market_a.estimated_rent_sqft_annual else rent_rate
    rent_rate_b = market_b.estimated_rent_sqft_annual if market_b and market_b.estimated_rent_sqft_annual else rent_rate

    inputs_a = _build_studio_inputs(vals_a, rent_rate_a)
    inputs_b = _build_studio_inputs(vals_b, rent_rate_b)

    if not inputs_a.city or not inputs_a.state or not inputs_b.city or not inputs_b.state:
        st.info("Enter **City** and **State** for both locations in the sidebar to get started.")
        st.stop()

    result_a = run_model(inputs_a, months=24, thresholds=thresholds)
    result_b = run_model(inputs_b, months=24, thresholds=thresholds)

    render_comparison_key_metrics(result_a, result_b, inputs_a, inputs_b)
    st.divider()

    render_comparison(result_a, result_b, inputs_a, inputs_b, market_a, market_b)
    st.divider()

    tab_a, tab_b = st.tabs([inputs_a.display_label or "Location A",
                            inputs_b.display_label or "Location B"])
    with tab_a:
        render_dashboard(inputs_a, result_a, market_a, thresholds, scope="loc_a")
    with tab_b:
        render_dashboard(inputs_b, result_b, market_b, thresholds, scope="loc_b")

# Fixed bottom-right footer — always visible
st.markdown(
    """
    <style>
    .fixed-footer {
        position: fixed;
        bottom: 12px;
        right: 18px;
        text-align: right;
        font-size: 11px;
        color: #9ca3af;
        line-height: 1.5;
        z-index: 9999;
        background: rgba(255,255,255,0.85);
        padding: 4px 10px;
        border-radius: 6px;
    }
    </style>
    <div class="fixed-footer">
        Studio Feasibility Calculator v7.0 -- For planning purposes only, not financial advice.<br>
        Built by Anmol Purohit
    </div>
    """,
    unsafe_allow_html=True,
)
