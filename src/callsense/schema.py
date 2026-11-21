from enum import Enum

from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    shipping = "shipping"
    cancellation = "cancellation"
    other = "other"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class CallAnalysis(BaseModel):
    """Structured analysis of one customer support call."""

    summary: str = Field(description="One or two sentences describing the call.")
    issue_category: IssueCategory = Field(
        description=(
            "Main reason the customer called. "
            "billing: charges, refunds, payments, declined cards, invoices. "
            "technical: something in the product is broken, erroring, or not working as expected. "
            "account: login, password, profile, or account access. "
            "shipping: delivery status, lost, late, damaged, or wrong items. "
            "cancellation: ending or downgrading a subscription or service. "
            "other: anything else, including how-to and feature questions where nothing is broken."
        )
    )
    customer_sentiment: Sentiment = Field(description="Customer's overall sentiment at the end of the call.")
    resolved: bool = Field(
        description=(
            "True if, by the end of the call, the customer's problem is fixed or the agent has completed every "
            "action needed to fix it (for example, issued a refund, shipped a replacement, cancelled a plan), "
            "even if automatic processing such as a refund posting or a package in transit still remains. "
            "False if fixing the problem still depends on investigation, escalation, or someone other than "
            "the agent acting (for example, the customer's bank or an engineering team)."
        )
    )
    follow_up_required: bool = Field(
        description=(
            "True if a person (the agent, another team, or the customer) must take a specific action after the "
            "call, such as returning an item, contacting their bank, or an engineer investigating. "
            "Automatic processes do not count: a refund posting, an automatic confirmation email, or a "
            "package already in transit is not follow-up."
        )
    )
    action_items: list[str] = Field(default_factory=list, description="Concrete follow-up actions, if any.")
    pii_mentioned: bool = Field(
        description=(
            "True if the call contains a complete, unmasked piece of personal data: card number, CVV, bank "
            "account number, SSN or other government ID, date of birth, home or delivery street address, phone number, or full "
            "email address. Masked or partial values (for example j.m****@example.com or a card's last four "
            "digits) do not count."
        )
    )
