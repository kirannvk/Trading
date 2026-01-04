from main import compute_lots_for_risk


def test_compute_lots_for_risk_positive():
    lots = compute_lots_for_risk(ce_entry=100, ce_stop=125, pe_entry=100, pe_stop=125,
                                 lot_size=25, capital=500000, per_trade_risk_pct=0.5)
    # risk per unit = 25+25=50; per lot=50*25=1250; allowed=2500 => 2 lots
    assert lots == 2


def test_compute_lots_for_risk_min_one():
    lots = compute_lots_for_risk(ce_entry=100, ce_stop=100, pe_entry=100, pe_stop=100,
                                 lot_size=25, capital=500000, per_trade_risk_pct=0.5)
    assert lots == 1

