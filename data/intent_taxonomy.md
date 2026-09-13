# Intent Taxonomy for Uber AI Support Agent

## Overview
This taxonomy defines **7 mutually exclusive operational intents** derived directly from the empirical analysis of 41,185 Uber Twitter customer-support conversations. Each intent represents a distinct functional pathway within Uber's customer experience and customer support infrastructure.

---

## Intent Summary Table

| # | Intent Name | Approx Dataset Share | Primary Resolution Action | Risk Level |
|---|---|---|---|---|
| 1 | `fare_and_billing_dispute` | ~20.9% | In-app Fare Review / Toll Audit | Medium |
| 2 | `cancellation_fee_issue` | ~8.4% | Cancellation Fee Waiver / Refund Flow | Low - Medium |
| 3 | `driver_behavior_and_safety` | ~19.1% | Human Escalation / Safety Incident Investigation | **High - Critical** |
| 4 | `lost_item_inquiry` | ~5.1% | Driver Contact Flow / In-App Calling | Medium |
| 5 | `pickup_and_route_issue` | ~3.7% | Route Map Audit / Inefficient Route Dispute | Low |
| 6 | `app_and_account_support` | ~14.2% | Account Recovery / 2FA / Promo Validation | Medium |
| 7 | `service_feedback_and_general` | ~28.6% | FAQ / Operating Cities / General Praise | Low |

---

## Detailed Intent Specifications

### 1. `fare_and_billing_dispute`
- **Description**: Customer disputes charges applied to their card, bank account, or digital wallet (e.g. Paytm, Apple Pay, PayPal), including upfront fare discrepancies, toll charges, or duplicate deductions.
- **Representative Real Examples**:
  - *"Why did Uber charge me 45 dollars for a 5 mile trip in normal traffic? Please refund this difference."*
  - *"I was charged twice on my credit card for the exact same ride from the airport!"*
  - *"Can you please refund the toll charge? The driver avoided the toll road completely."*
- **Edge Cases**:
  - Customer asks for an invoice/receipt without disputing the amount (handled as routine self-service).
  - High dollar amount disputes (> $100) or allegations of credit card fraud (escalated to human billing audit).

### 2. `cancellation_fee_issue`
- **Description**: Inquiries or waiver requests regarding fees charged when a ride was cancelled by either the rider or the driver.
- **Representative Real Examples**:
  - *"Driver called me and asked me to cancel the trip because he didn't want to go south, and I got charged $5 fee!"*
  - *"I cancelled within 45 seconds of requesting because ETA was too long, why was I charged cancellation fee?"*
  - *"Driver was sitting at the same spot for 15 minutes and never came, so I had to cancel. Refund my fee!"*
- **Edge Cases**:
  - Driver asked rider to cancel vs rider cancelling after waiting beyond the estimated arrival window.
  - Driver repeatedly accepts and cancels, creating multiple fees (requires human review).

### 3. `driver_behavior_and_safety`
- **Description**: Grievances involving driver conduct, dangerous/reckless driving, traffic violations, verbal abuse, suspected intoxication, or physical security concerns.
- **Representative Real Examples**:
  - *"Driver was driving 90mph on the highway in heavy rain and weaving between trucks, I felt terrified."*
  - *"The driver started shouting obscenities at me when I asked him to turn down the radio."*
  - *"I suspect my driver was under the influence of alcohol, he was slurring his words."*
  - *"Driver refused to drop me at my doorstep late at night and told me to get out on the main road."*
- **Edge Cases**:
  - Mild driver complaints (e.g. car smelled like smoke or driver did not speak English) vs emergency safety hazards (threats, assault). Both escalate to humans, but emergencies receive top priority.

### 4. `lost_item_inquiry`
- **Description**: Customer forgot or left a personal possession (smartphone, keys, bag, wallet, passport) inside a vehicle and needs to contact the driver.
- **Representative Real Examples**:
  - *"I accidentally left my iPhone 13 in the back seat of the Honda Civic that dropped me at 8 PM."*
  - *"Left my brown leather wallet containing my driver license and credit cards in the car!"*
  - *"Can Uber provide me the direct phone number of the driver? I left my house keys in the cab."*
- **Edge Cases**:
  - Time-critical items (e.g. passport 2 hours before an international flight) -> Escalated immediately to urgent dispatch.
  - Driver is unresponsive to in-app contact attempts for over 24 hours -> Escalated to human support.

### 5. `pickup_and_route_issue`
- **Description**: Inaccuracies in driver arrival, GPS pickup pin placement, driver refusing to navigate to destination, or driver taking a circuitous detour.
- **Representative Real Examples**:
  - *"Driver took an absurdly long detour through heavy city traffic adding 30 minutes to the trip."*
  - *"The pickup pin placed the driver 3 blocks away across a closed highway barrier."*
  - *"Driver went the opposite direction on the one-way street, GPS in app is completely broken."*
- **Edge Cases**:
  - Route issues that caused an inflated fare are closely tied with `fare_and_billing_dispute`. Triage routes to fare review if money is requested, or route feedback if navigation is the primary complaint.

### 6. `app_and_account_support`
- **Description**: Technical issues with the mobile application, account login errors, 2FA verification SMS, promo code failures, or deactivation inquiries.
- **Representative Real Examples**:
  - *"My account was deactivated suddenly and I have no idea why. Please reactivate my account!"*
  - *"I changed my phone number and now cannot receive the 2FA SMS code to sign in."*
  - *"Entered promo code SAVE50 before booking but the discount was not applied to my receipt."*
- **Edge Cases**:
  - Account hacked or unauthorized device login -> Triggers immediate security escalation.
  - Expired vs invalid promo codes.

### 7. `service_feedback_and_general`
- **Description**: General customer comments, operating hours/cities, policy questions, complimenting driver service, or general inquiries.
- **Representative Real Examples**:
  - *"Is Uber service available in Honolulu Hawaii for early morning airport pickups?"*
  - *"Shoutout to driver Marcus in Chicago, he was courteous and returned my umbrella. Great service!"*
  - *"What is your policy on bringing service animals into vehicles?"*
- **Edge Cases**:
  - Broad compliments that don't require an action versus multi-intent broad complaints.
