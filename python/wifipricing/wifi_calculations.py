def get_profit(revenue, data_mb, cost_per_mb = 0.05):
    """Calculate profit from revenue and data usage"""
    try:
        prof = revenue - data_mb * cost_per_mb
    except:
        prof = None

    return prof