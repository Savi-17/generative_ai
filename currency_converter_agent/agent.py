from google.adk.agents import Agent

def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method."""
    fee_database = {
        "platinum credit card": 0.02,
        "gold debit card": 0.035,
        "bank transfer": 0.01,
    }
    fee = fee_database.get(method.lower())
    if fee is not None:
        return {"status": "success", "fee_percentage": fee}
    else:
        return {"status": "error", "error_message": f"Payment method '{method}' not found"}

def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies."""
    rate_database = {
        "usd": {"eur": 0.93, "jpy": 157.5, "inr": 83.58},
    }
    rate = rate_database.get(base_currency.lower(), {}).get(target_currency.lower())
    if rate is not None:
        return {"status": "success", "rate": rate}
    else:
        return {"status": "error", "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}"}

root_agent = Agent(
    name="currency_converter_agent",
    model="gemini-2.0-flash",
    description="Converts currencies and applies transaction fees.",
    instruction="You convert currencies using get_fee_for_payment_method() and get_exchange_rate() tools.",
    tools=[get_fee_for_payment_method, get_exchange_rate],
)
