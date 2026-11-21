"""The fixed seed list of call scenarios for Kestrel Home, a fictional smart-home company.

Every synthetic call is one of these scenarios. The code decides the scenario, outcome and mood of
each call; Claude only writes the dialogue. That keeps the trends in the data under our control.
"""

from dataclasses import dataclass

COMPANY_BRIEF = """Kestrel Home is a fictional smart-home company. Its products:
- Kestrel Cam: indoor and outdoor security camera
- Kestrel Doorbell: video doorbell
- Kestrel Hub: the base station the cameras and doorbell connect to
- Kestrel Plus: subscription for cloud video recording, $9.99 a month or $99 a year
- the Kestrel app, for iOS and Android"""

PRODUCTS = ("Kestrel Cam", "Kestrel Doorbell", "Kestrel Hub", "Kestrel Plus", "Kestrel app")
DEVICES = ("Kestrel Cam", "Kestrel Doorbell", "Kestrel Hub")

REGIONS = ("Northeast", "Southeast", "Midwest", "West")
REGION_WEIGHTS = (0.30, 0.25, 0.20, 0.25)

# Share of ordinary (non-event) calls in each category. Matches IssueCategory in schema.py.
CATEGORY_WEIGHTS = {
    "billing": 0.24,
    "technical": 0.24,
    "account": 0.14,
    "shipping": 0.14,
    "cancellation": 0.10,
    "other": 0.14,
}


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str  # the issue_category a correct analysis should give
    situation: str  # why the customer is calling
    resolved_outcome: str  # how the call ends when the agent fixes it
    unresolved_outcome: str  # how the call ends when it can't be fixed on the call
    products: tuple[str, ...]
    resolve_rate: float  # share of calls that end resolved, before any planted event
    pii_kinds: tuple[str, ...] = ()  # personal data that could plausibly come up
    event_only: bool = False  # only appears through a planted event, never in the ordinary mix
    weight: float = 1.0  # how often it's picked relative to the other scenarios in its category


SCENARIOS = [
    # billing
    Scenario(
        "billing_double_charge", "billing",
        "The customer was charged twice for the same Kestrel Plus renewal.",
        "The agent confirms the duplicate and refunds it on the call; it will post in 3 to 5 business days.",
        "The agent can only see one charge and opens a billing investigation; the billing team will reply by email.",
        ("Kestrel Plus",), 0.8, ("CARD_NUMBER", "EMAIL"),
    ),
    Scenario(
        "billing_unexpected_renewal", "billing",
        "The customer didn't expect their annual Kestrel Plus plan to renew and wants to understand the charge.",
        "The agent explains the renewal, the charge is within 30 days, so the agent refunds it and turns off auto-renew.",
        "The agent explains the renewal but needs a supervisor to approve the refund; the supervisor will call back.",
        ("Kestrel Plus",), 0.7, ("EMAIL", "PHONE"),
    ),
    Scenario(
        "billing_refund_request", "billing",
        "The customer bought a device or subscription a few days ago and wants a refund.",
        "The agent confirms the purchase is within the refund window and issues the refund on the call.",
        "The agent needs the customer to send the device back first and emails a return label.",
        ("Kestrel Cam", "Kestrel Doorbell", "Kestrel Hub", "Kestrel Plus"), 0.75, ("EMAIL", "ADDRESS"),
        weight=3.0,  # common enough that the refund policy change has a measurable sample
    ),
    Scenario(
        "billing_card_declined", "billing",
        "The customer's card was declined when their subscription tried to renew.",
        "The customer gives a new card and the agent takes the payment successfully on the call.",
        "The card issuer is blocking the payment; the customer has to call their bank.",
        ("Kestrel Plus",), 0.6, ("CARD_NUMBER",),
    ),
    Scenario(
        "billing_invoice_question", "billing",
        "The customer needs a copy of an invoice for an expense report and asks about a line item.",
        "The agent explains the line item and emails the invoice right away.",
        "The invoice system is down; the agent promises to email the invoice once it's back.",
        ("Kestrel Plus", "Kestrel Cam", "Kestrel Doorbell"), 0.85, ("EMAIL",),
    ),
    # technical
    Scenario(
        "tech_camera_offline", "technical",
        "The Kestrel Cam keeps going offline every few hours.",
        "The agent walks the customer through moving the hub and changing the Wi-Fi channel, and the camera stays online.",
        "Troubleshooting doesn't help; the agent escalates to the engineering team, who will contact the customer.",
        ("Kestrel Cam",), 0.6,
    ),
    Scenario(
        "tech_hub_wont_pair", "technical",
        "A new device won't pair with the Kestrel Hub.",
        "The agent has the customer reset the device and re-pair it, and it works.",
        "Pairing still fails; the agent arranges a replacement hub to be shipped once the old one is returned.",
        ("Kestrel Hub", "Kestrel Cam", "Kestrel Doorbell"), 0.65, ("ADDRESS",),
    ),
    Scenario(
        "tech_app_crash", "technical",
        "The Kestrel app crashes when the customer opens the video history screen.",
        "Reinstalling the app and clearing its cache fixes the crash.",
        "The crash continues; the agent files a bug report and the app team will follow up.",
        ("Kestrel app",), 0.55,
    ),
    Scenario(
        "tech_firmware_update_failed", "technical",
        "A firmware update failed and the device's light is blinking red.",
        "The agent guides a manual firmware recovery and the device comes back.",
        "Recovery fails; the device needs to be replaced and the customer must ship it back first.",
        ("Kestrel Cam", "Kestrel Doorbell", "Kestrel Hub"), 0.6, ("ADDRESS",),
    ),
    Scenario(
        "tech_motion_alerts_late", "technical",
        "Motion alerts arrive several minutes late on the customer's phone.",
        "The agent finds battery-saver mode is delaying notifications and has the customer turn it off for the app.",
        "Settings look correct; the agent escalates to engineering to look at the notification logs.",
        ("Kestrel Cam", "Kestrel Doorbell", "Kestrel app"), 0.65,
    ),
    # account
    Scenario(
        "account_password_reset", "account",
        "The customer forgot their password and the reset email isn't arriving.",
        "The agent finds the email in a spam filter rule, resends it, and the customer resets the password on the call.",
        "The reset email still doesn't arrive; the agent escalates to the accounts team.",
        ("Kestrel app",), 0.85, ("EMAIL",),
    ),
    Scenario(
        "account_email_change", "account",
        "The customer wants to change the email address on their account.",
        "The agent verifies the customer's identity and updates the email on the call.",
        "The customer can't pass identity verification; they need to send ID to the accounts team.",
        ("Kestrel app",), 0.8, ("EMAIL", "DATE_OF_BIRTH", "PHONE"),
    ),
    Scenario(
        "account_2fa_locked_out", "account",
        "The customer got a new phone and is locked out by two-factor authentication.",
        "The agent verifies identity and resets two-factor so the customer can sign in again.",
        "The agent can't verify identity on the call; the accounts team will review a recovery request within 48 hours.",
        ("Kestrel app",), 0.65, ("DATE_OF_BIRTH", "PHONE"),
    ),
    Scenario(
        "account_shared_access", "account",
        "The customer wants to give a family member access to the cameras.",
        "The agent explains how to send a shared-access invite and the family member accepts it during the call.",
        "The invite feature errors out; the agent escalates it to the accounts team.",
        ("Kestrel app",), 0.85, ("EMAIL",),
    ),
    # shipping
    Scenario(
        "ship_late_delivery", "shipping",
        "An order is several days past its estimated delivery date.",
        "The agent sees the package was lost and ships a replacement by express on the call.",
        "The package is stuck at the carrier; the agent opens a trace with the carrier, who will update the customer.",
        DEVICES, 0.6, ("ADDRESS", "PHONE"),
    ),
    Scenario(
        "ship_damaged_item", "shipping",
        "The device arrived with a cracked case.",
        "The agent ships a replacement right away and emails a prepaid label for the damaged one.",
        "The agent needs photos of the damage before a replacement can be approved.",
        DEVICES, 0.75, ("ADDRESS", "EMAIL"),
    ),
    Scenario(
        "ship_wrong_item", "shipping",
        "The customer received the wrong product.",
        "The agent ships the correct product and emails a return label for the wrong one.",
        "The correct product is out of stock; the agent puts the customer on a waitlist.",
        DEVICES, 0.7, ("ADDRESS",),
    ),
    Scenario(
        "ship_address_change", "shipping",
        "The customer moved and wants to change the delivery address on an order that hasn't shipped.",
        "The agent updates the address before the order ships.",
        "The order has already left the warehouse; the customer has to ask the carrier to redirect it.",
        DEVICES, 0.7, ("ADDRESS", "PHONE"),
    ),
    # cancellation
    Scenario(
        "cancel_too_expensive", "cancellation",
        "The customer wants to cancel Kestrel Plus because it's too expensive.",
        "The agent cancels the subscription effective at the end of the billing period.",
        "The agent can't process the cancellation because of a system error and asks the customer to call back tomorrow.",
        ("Kestrel Plus",), 0.9, ("EMAIL",),
    ),
    Scenario(
        "cancel_moving", "cancellation",
        "The customer is moving abroad and wants to cancel their subscription.",
        "The agent cancels the subscription and confirms no further charges.",
        "The account has an unpaid balance that must be settled before the cancellation can go through.",
        ("Kestrel Plus",), 0.85,
    ),
    Scenario(
        "cancel_not_using", "cancellation",
        "The customer barely uses the cameras and wants to downgrade from the annual plan to monthly.",
        "The agent switches the plan to monthly from the next renewal.",
        "Downgrades mid-term aren't possible; the customer has to wait until renewal and call again.",
        ("Kestrel Plus",), 0.8,
    ),
    Scenario(
        "cancel_competitor", "cancellation",
        "The customer is switching to a competitor's cameras and wants to cancel.",
        "The agent offers a discount, the customer declines, and the agent cancels the plan.",
        "The customer wants a prorated refund too, which needs a supervisor's approval; the supervisor will call back.",
        ("Kestrel Plus",), 0.8, ("PHONE",),
    ),
    # other
    Scenario(
        "other_recording_schedule", "other",
        "The customer asks how to set the camera to record only at night.",
        "The agent walks them through setting up a schedule in the app.",
        "The schedule option the customer wants doesn't exist yet; the agent notes it as feedback.",
        ("Kestrel Cam", "Kestrel app"), 0.9,
    ),
    Scenario(
        "other_compatibility", "other",
        "The customer asks whether the doorbell works with their smart speaker.",
        "The agent explains which speakers are supported and how to link them.",
        "The agent isn't sure and says the product team will email an answer.",
        ("Kestrel Doorbell", "Kestrel Hub"), 0.85, ("EMAIL",),
    ),
    Scenario(
        "other_feature_question", "other",
        "The customer asks what Kestrel Plus includes compared with the free plan.",
        "The agent explains the differences clearly.",
        "The agent can't answer a question about video storage limits and promises an email with details.",
        ("Kestrel Plus",), 0.9,
    ),
    # only through planted events
    Scenario(
        "billing_payment_failed_outage", "billing",
        "The customer's payment keeps failing at checkout, or they were charged but got no order confirmation.",
        "The agent places the order manually over the phone and it goes through.",
        "The agent says the payment processor is having an outage, engineers are working on it, and the customer "
        "should try again later; any duplicate charge will be reversed once it's fixed.",
        ("Kestrel Plus", "Kestrel Cam", "Kestrel Doorbell"), 0.15, ("CARD_NUMBER", "EMAIL"), event_only=True,
    ),
    Scenario(
        "account_app_login_loop", "account",
        "Since updating the Kestrel app to version 5.2, the customer can't log in: after entering their password, "
        "the app goes back to the login screen.",
        "The agent has the customer sign in on the website first and then in the app, which works.",
        "Nothing works; the agent says other customers have reported the same login problem after the update and "
        "the app team is investigating.",
        ("Kestrel app",), 0.3, ("EMAIL",), event_only=True,
    ),
    Scenario(
        "ship_doorbell_backorder", "shipping",
        "The customer's Kestrel Doorbell order hasn't shipped and is a week late.",
        "The agent upgrades the order to express shipping from another warehouse, and it ships today.",
        "The agent explains that doorbells are delayed by a carrier backlog at the central warehouse and gives a new "
        "estimate of 7 to 10 days.",
        ("Kestrel Doorbell",), 0.2, ("ADDRESS", "PHONE"), event_only=True,
    ),
]

SCENARIOS_BY_ID = {s.id: s for s in SCENARIOS}

# Endings used by planted events instead of the scenario's own.
REFUND_DENIED_OUTCOME = (
    "The agent explains that since July 27 the refund window is 14 days instead of 30, the purchase is outside it, "
    "and refuses the refund. The customer is not happy about the policy change."
)
MIDWEST_ESCALATION_OUTCOME = (
    "The agent says they don't have the system access to do this themselves and must pass it to a back-office team, "
    "which will call the customer back in 3 to 5 business days."
)

MOODS = {
    "positive": "pleased: in their last turns they thank the agent warmly or say they're relieved or happy",
    "neutral": "calm: in their last turns they accept the outcome matter-of-factly, polite but not enthusiastic",
    "negative": "frustrated: in their last turns they complain, sound annoyed, or accept the outcome grudgingly",
}

SMALL_TALK_TOPICS = (
    "the weather", "their dog barking in the background", "weekend plans", "a local sports team",
    "the hold music", "a long day at work", "a recent holiday",
)

FIRST_NAMES = (
    "Maya", "Daniel", "Priya", "Marcus", "Elena", "Tom", "Aisha", "Kevin", "Sofia", "James", "Lena", "Omar",
    "Grace", "Luis", "Hannah", "Ravi", "Chloe", "Ben", "Nina", "Sam", "Fatima", "Jake", "Mei", "Carlos",
)
LAST_NAMES = (
    "chen", "patel", "garcia", "nguyen", "smith", "kowalski", "okafor", "rossi", "kim", "brennan", "silva", "haddad",
)
AGENT_NAMES = ("Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Drew")
STREETS = ("Maple", "Oak", "Cedar", "Willow", "Lakeview", "Hillcrest", "Birch", "Sunset", "Park", "River")
STREET_SUFFIXES = ("Street", "Avenue", "Road", "Lane", "Drive", "Court")
AREA_CODES = ("415", "312", "617", "206", "404", "512", "303", "612")
# Card networks' published test numbers: they pass the checksum but can't be charged.
TEST_CARDS = ("4111 1111 1111 1111", "4242 4242 4242 4242", "5555 5555 5555 4444", "4012 8888 8888 1881")
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
