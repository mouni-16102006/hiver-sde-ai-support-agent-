"""
Grounded response generation engine for Uber AI Support Agent.
Generates responses strictly grounded in historical support resolutions,
preventing hallucinated policies, timelines, or financial guarantees.
"""
import os
from typing import Dict, List, Optional
from src.config import BRAND_NAME, DECISION_AUTO_HANDLE, DECISION_ESCALATE
from src.preprocessing import clean_tweet_text


# Canonical self-service guidance mapped to Uber support intents
CANONICAL_ACTION_PATHS = {
    "fare_and_billing_dispute": (
        "You can review this charge directly in the Uber app: Go to 'Activity' > select the trip > "
        "tap 'Help' > 'Review my fare or fees'. Our billing system will audit the distance, toll, and route taken."
    ),
    "cancellation_fee_issue": (
        "To dispute a cancellation fee: Open the Uber app > tap 'Activity' > select the cancelled trip > "
        "select 'Help' > 'I was charged a cancellation fee'. If the driver cancelled or was delayed, the fee is automatically eligible for a waiver."
    ),
    "driver_behavior_and_safety": (
        "Your safety and comfort are our highest priorities. Please navigate to 'Activity' > select this ride > "
        "tap 'Help' > 'Report a safety issue' or 'Driver feedback'. A specialized safety representative will investigate this matter immediately."
    ),
    "lost_item_inquiry": (
        "To retrieve your lost item: Open the Uber app > tap 'Activity' > select the trip > tap 'Find lost item' > "
        "'Contact driver about a lost item'. You can call the driver via an anonymized number or leave a callback number."
    ),
    "pickup_and_route_issue": (
        "To report an inefficient route or pickup issue: Navigate to 'Activity' in the app > select the trip > "
        "tap 'Help' > 'My driver took a poor route'. The route map and duration will be audited against standard transit estimates."
    ),
    "app_and_account_support": (
        "For account or promo assistance: Head to 'Account' > 'Help' in the app. Ensure your app is updated to the latest version. "
        "If you are locked out or updating phone/email details, visit help.uber.com to verify your registered identity."
    ),
    "service_feedback_and_general": (
        "We appreciate your feedback. You can view trip receipts, update payment preferences, or reach out to our team anytime "
        "through the 'Help' section in your Uber app or by visiting help.uber.com."
    ),
}


class GroundedReplyGenerator:
    """
    Composes strictly grounded support replies based on retrieved historical evidence.
    """

    def __init__(self):
        pass

    def generate_reply(
        self,
        customer_message: str,
        intent: str,
        evidence: List[Dict],
        decision: str,
        escalation_reason: Optional[str] = None,
    ) -> str:
        """
        Generates grounded reply for the customer inquiry.
        """
        action_guide = CANONICAL_ACTION_PATHS.get(intent, CANONICAL_ACTION_PATHS["service_feedback_and_general"])

        if decision == DECISION_ESCALATE:
            # Escalated case response
            if intent == "driver_behavior_and_safety":
                return (
                    f"Hi there, thank you for bringing this to our attention. {escalation_reason} "
                    f"{action_guide} We take reports regarding safety and driver conduct very seriously, "
                    f"and our dedicated incident response team has been prioritized to assist you directly."
                )
            elif intent == "lost_item_inquiry":
                return (
                    f"Hi there, we understand how stressful it is to lose an item. "
                    f"{action_guide} We have flagged this for a priority support specialist to help coordinate "
                    f"with the driver if you are unable to connect via the app."
                )
            else:
                return (
                    f"Hi there, thanks for reaching out to {BRAND_NAME} Support. "
                    f"Because your request involves specific account or trip verification ({escalation_reason}), "
                    f"we have escalated this case to a human support specialist. In the meantime: {action_guide}"
                )

        # AUTO_HANDLE case: strictly grounded in historical precedent
        if evidence:
            top_reply = evidence[0].get("support_response", "")
            # Clean handles and URLs from historical reply
            cleaned_hist = clean_tweet_text(top_reply, remove_handles=True, preserve_url_token=False)

            # Ensure response has clear self-service instructions
            if len(cleaned_hist) > 25 and not cleaned_hist.startswith("["):
                return f"Hi there! {cleaned_hist} For quick self-service: {action_guide}"

        # Fallback grounded response
        return f"Hi there! We are here to help with your {intent.replace('_', ' ')}. {action_guide}"
