#!/usr/bin/env python3
"""
app.py – CLI entry point for Studio Feasibility Calculator.

Usage:
    python app.py --location "Bellevue, WA" --sqft 2500 --arpm 359.40 --rent 15205 --capex 150000
    python app.py --location "Bellevue, WA" --arpm 359.40 --rent 15205 --compare "Redmond, WA" --rent2 13080
    python app.py --help
"""

from __future__ import annotations

import argparse
import json
import sys

from market_research import research_location
from models import FixedCosts, StudioInputs, VariableCosts, run_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="New Studio Feasibility Calculator – CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--location", required=True, help='City, State (e.g. "Bellevue, WA")')
    p.add_argument("--sqft", type=int, default=2500)
    p.add_argument("--arpm", type=float, default=359.40, help="ARPM Year 1 ($)")
    p.add_argument("--arpm-y2", type=float, default=None, help="ARPM Year 2 ($), if different")
    p.add_argument("--target-members", type=int, default=250)
    p.add_argument("--capex", type=float, default=150_000)
    p.add_argument("--opening-costs", type=float, default=15_000)
    p.add_argument("--rent", type=float, default=0, help="Monthly rent ($); 0 = auto-estimate")
    p.add_argument("--marketing", type=float, default=3_000)
    p.add_argument("--net-adds", type=int, default=15, help="Net new members per month")
    p.add_argument("--trainer-rate", type=float, default=35.0, help="Trainer $/class hr")
    p.add_argument("--classes-week", type=int, default=30)
    p.add_argument("--scenario", choices=["downside", "base", "upside"], default="base")
    p.add_argument("--months", type=int, default=24)
    p.add_argument("--research", action="store_true", help="Run market research")
    p.add_argument("--json-out", type=str, default=None)
    # Comparison
    p.add_argument("--compare", type=str, default=None, help='Second location: "Redmond, WA"')
    p.add_argument("--rent2", type=float, default=0, help="Rent for second location")
    return p.parse_args()


def _parse_location(loc: str):
    parts = [s.strip() for s in loc.split(",")]
    return parts[0] if parts else "Unknown", parts[1] if len(parts) > 1 else ""


def _build_inputs(city, state, args, rent_override=None) -> StudioInputs:
    rent = rent_override if rent_override and rent_override > 0 else args.rent
    fc = FixedCosts(rent=rent, marketing=args.marketing)
    vc = VariableCosts(trainer_hourly_rate=args.trainer_rate, classes_per_week=args.classes_week)
    return StudioInputs(
        city=city, state=state, sqft=args.sqft,
        arpm=args.arpm, arpm_year2=args.arpm_y2,
        target_members=args.target_members,
        monthly_net_adds=args.net_adds,
        capex=args.capex, opening_costs=args.opening_costs,
        fixed_costs=fc, variable_costs=vc,
        scenario=args.scenario,
    )


def _print_result(result, city, state, scenario):
    light_icons = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    print(f"\n{'=' * 60}")
    print(f"  STUDIO FEASIBILITY – {city}, {state}")
    print(f"  Scenario: {scenario.upper()}")
    print(f"{'=' * 60}")
    print(f"  {light_icons.get(result.traffic_light, '⚪')}  Score: {result.feasibility_score:.0f}/100")
    print(f"  Break-even members : {result.break_even_members or 'N/A'}")
    print(f"  Break-even month   : {result.break_even_month or 'N/A'}")
    print(f"  Payback period     : {result.payback_months or 'N/A'} months")
    print(f"  1-Year ROI         : {result.roi_1y:.1f}%" if result.roi_1y else "  1-Year ROI         : N/A")
    print(f"  3-Year ROI         : {result.roi_3y:.1f}%" if result.roi_3y else "  3-Year ROI         : N/A")
    print(f"  IRR (annualised)   : {result.irr:.1f}%" if result.irr else "  IRR                : N/A")
    print(f"  Month 24 EBITDA    : ${result.pro_forma[-1].ebitda:,.0f}" if result.pro_forma else "")
    print(f"{'=' * 60}")

    print(f"\n  Mo | Members | ARPM    |   Revenue |  Tot OpEx |    EBITDA |   Cum CF")
    print(f"  {'-' * 75}")
    for r in result.pro_forma:
        print(f"  {r.month:>2} | {r.members:>7,} | ${r.arpm_used:>6,.2f} | ${r.revenue_total:>8,.0f} "
              f"| ${r.total_opex:>8,.0f} | ${r.ebitda:>8,.0f} | ${r.cumulative_cf:>8,.0f}")


def main() -> None:
    args = parse_args()
    city, state = _parse_location(args.location)

    if args.research:
        print(f"\n🔍 Market research for {city}, {state}...")
        market = research_location(city, state)
        print(f"   Income: ${market.median_household_income:,.0f}  ({market.income_source})")
        print(f"   Density: {market.population_density:,.0f}/sq mi" if market.population_density else "")
        print(f"   Competitors: {market.competition_count}")
        print(f"   Suggested ARPM: ${market.suggested_arpm_low:.0f}-${market.suggested_arpm_high:.0f}")

    inputs = _build_inputs(city, state, args)
    result = run_model(inputs, months=args.months)
    _print_result(result, city, state, args.scenario)

    # Comparison mode
    if args.compare:
        city2, state2 = _parse_location(args.compare)
        inputs2 = _build_inputs(city2, state2, args, rent_override=args.rent2)
        result2 = run_model(inputs2, months=args.months)
        _print_result(result2, city2, state2, args.scenario)

        print(f"\n{'=' * 60}")
        print(f"  COMPARISON: {city} vs {city2}")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<25s} {'  ' + city:<15s} {'  ' + city2:<15s} {'Better'}")
        print(f"  {'-' * 65}")
        comparisons = [
            ("Score", result.feasibility_score, result2.feasibility_score, True),
            ("Payback (mo)", result.payback_months, result2.payback_months, False),
            ("BE Month", result.break_even_month, result2.break_even_month, False),
            ("1Y ROI %", result.roi_1y, result2.roi_1y, True),
            ("M24 EBITDA", result.pro_forma[-1].ebitda, result2.pro_forma[-1].ebitda, True),
        ]
        for name, v1, v2, higher in comparisons:
            s1 = f"{v1}" if v1 is not None else "N/A"
            s2 = f"{v2}" if v2 is not None else "N/A"
            better = ""
            if v1 is not None and v2 is not None:
                if higher:
                    better = city if v1 > v2 else city2 if v2 > v1 else "Tie"
                else:
                    better = city if v1 < v2 else city2 if v2 < v1 else "Tie"
            print(f"  {name:<25s} {s1:>13s} {s2:>13s}   {better}")

    if args.json_out:
        exp = {
            "location": f"{city}, {state}",
            "results": {
                "be_members": result.break_even_members,
                "be_month": result.break_even_month,
                "payback": result.payback_months,
                "roi_1y": result.roi_1y, "roi_3y": result.roi_3y,
                "irr": result.irr, "score": result.feasibility_score,
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(exp, f, indent=2, default=str)
        print(f"\n  ✅ JSON → {args.json_out}")


if __name__ == "__main__":
    main()
