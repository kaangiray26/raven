# Prices for 1K tokens
costs = {
    "gpt-4o": 0.0025,
    "gpt-4o-mini": 0.00015,
    "gpt-4o-mini-2024-07-18": 0.00015,
    "gpt-4-turbo": 0.01,
    "gpt-4": 0.03,
    "gpt-3.5-turbo": 0.0005,
    "text-embedding-3-small": 0.000020,
    "text-embedding-3-large": 0.000130,
    "ada v2": 0.000100,
}

def pricing_for_model(model, tokens):
    if model not in costs:
        return 0
    return round(costs[model] * (tokens / 1000), 8)

def pricing_for_usage(models: dict):
    usage = {}
    for model, tokens in models.items():
        usage[model] = "${:.8f}".format(pricing_for_model(model, tokens))
    return usage