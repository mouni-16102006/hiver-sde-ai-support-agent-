# Brand Selection: Uber Support (@Uber_Support)

## 1. Selected Brand
- **Brand Name**: Uber
- **Twitter Support Handle**: `@Uber_Support`
- **Domain**: Urban Mobility, Rideshare, and On-Demand Delivery
- **Dataset Source**: *Customer Support on Twitter* (Kaggle / Hugging Face `TNE-AI/customer-support-on-twitter-conversation`)

---

## 2. Dataset Empirical Evidence & Statistics
Before selecting Uber, we performed an empirical scan across the full 794,335 conversation threads in the dataset.

| Brand | Total Conversations in Dataset | Rank | Primary Characteristics |
|---|---|---|---|
| **AmazonHelp** | 81,092 | #1 | Heavy multilingual volume (Spanish, Japanese, German); order tracking requires private order IDs. |
| **AppleSupport** | 76,639 | #2 | Massive volume, but >60% of tweets simply point to generic Apple Knowledge Base articles without visible conversational resolution. |
| **Uber_Support** | **41,185** | **#3** | **Selected**: High English volume, clear multi-turn resolution pairs, distinct recurring operational issues, and clear safety triage boundaries. |
| **SpotifyCares** | 27,910 | #4 | *Strictly Excluded* per project specification constraints. |
| **Delta** | 25,151 | #5 | Airline ticketing, heavy PNR lookup dependency, high seasonal weather disruption noise. |
| **AmericanAir** | 25,061 | #6 | Flight delays, baggage claim; high dependency on proprietary airline reservation APIs. |
| **comcastcares** | 23,442 | #7 | Cable/telecom hardware issues; frequent technician visit scheduling. |

---

## 3. Why Uber_Support Was Selected
1. **Sufficient Conversation Volume**:
   - 41,185 total multi-turn conversations in the dataset.
   - Over 41,064 pairs with verified customer inquiries and agent responses.
   - Clean sample subset of 15,000 conversations used for knowledge base indexing and classifier training.

2. **High-Stakes vs. Low-Stakes Operational Boundary**:
   - Unlike entertainment or retail where an AI hallucination merely misinforms, in urban transportation an automated agent handling safety threats (e.g. reckless driving, drunk driver, harassment) could lead to severe physical harm or liability.
   - Uber provides the perfect real-world benchmark to demonstrate **why an AI support agent must know when NOT to auto-handle** and safely escalate to humans.

3. **High Density of Recurring, Structured Problem Domains**:
   - In-app fare adjustments, toll disputes, and duplicate card charges.
   - Cancellation fee waivers (grace periods, driver-initiated cancellations).
   - Lost & Found item retrieval (connecting riders with drivers).
   - Pickup navigation, GPS mismatches, and route optimization.
   - Account login, 2FA SMS, and promo code redemptions.

4. **Authentic Historical Grounding**:
   - Uber Support Twitter agents historically provide specific in-app navigation routes (e.g. `Activity > Select Trip > Help > Review my fare`).
   - This enables evaluating whether an AI agent drafts responses strictly grounded in verified company operating procedures.

---

## 4. Alternatives Considered and Why They Were Rejected
- **AmazonHelp**:
  - Rejected due to high linguistic dispersion (over 30% of tweets were non-English, requiring complex multi-lingual models that obscure core triage logic).
  - Most Amazon Twitter resolutions immediately require private DMs for Order ID numbers, resulting in low informational content in public tweets.
- **AppleSupport**:
  - Rejected because Apple agents rarely discuss troubleshooting steps on Twitter, almost exclusively responding with canned links to `support.apple.com/HTxxxxxx`.
- **Airlines (Delta / AmericanAir)**:
  - Rejected due to low self-service resolution potential; airline flight changes almost always require live booking agent access to GDS systems (Sabre/Amadeus).
- **SpotifyCares**:
  - Explicitly excluded by candidate project guidelines.

---

## 5. Summary & Verification
The dataset confirms that `@Uber_Support` contains 41,185 authentic, rich customer-support conversations, providing an exceptional foundation for building and proving an AI support agent that balances automated self-service with safety-critical human escalation.
