# 🍔 Is This Login... *Teddy?* 👷‍♂️

Is this login weird?  
Behavioral login anomaly detection, Bob's Burgers–style.

## What this is

Bob runs a restaurant.  
He knows his regulars.

This project does the same thing—but for logins.

Instead of asking  
**"Is this malicious?"**. 
it asks  
**"Does this feel like someone who's usually here?"**

The system learns what *normal* login behavior looks like, then points at the stuff that makes Bob squint across the counter.

Small. Readable. Defensive mindset first.

## Why this exists

Rule-based auth detection is like Bob writing rules on the wall:

* "No yelling"
* "No loitering"
* "No coming back 12 times an hour"

It works for about five minutes.

Then:

* Attackers adapt
* Rules get weirdly specific
* Edge cases pile up like unwashed dishes

Machine learning helps by learning **patterns**, not commandments:

* When people usually log in
* Where they usually come from
* What devices they usually use
* How fast they usually move

When something *breaks the vibe*, it gets flagged.

## What the model looks at

Think of these as the things Bob notices without trying:

* `hour_of_day` — why are you here at 3am
* `country` — are you... local
* `device_type` — phone, desktop, or Linda-on-a-tablet
* `login_success` — do you know your password or not
* `sessions_per_hour` — why are you still here

No PII.  
No payload inspection.  
Just behavior.

Bob doesn't need your life story — he needs to know if this is normal.

## How it works (high level)

1. Logins walk into the restaurant
2. The model learns what "regular customers" look like
3. New logins get scored by how fast they stand out
4. The weird ones get highlighted for review

Models explored:

* **Isolation Forest** (aka: *How fast would Bob realize this is Teddy?*)

## What this is *not*

* ❌ A production auth system
* ❌ A magic "catch all attackers" button
* ❌ A giant deep learning monster

This is a **thinking tool**.

For blue teamers who want to automate their gut feeling without turning it into 400 brittle rules.

## The question this actually answers

**"Would Bob trust this customer?"**

If the model hesitates, you probably should too.

Flag ≠ block.  
Weird ≠ malicious.

Sometimes it's just Teddy.  
Sometimes it's not.

## Future ideas

* k-means clustering (grouping regulars vs *characters*)
* simple statistical baselines (Bob's common sense)
* visualizing "normal" vs "what are you doing here"

## Disclaimer

This project uses **synthetic or anonymized data only**.

No real credentials.  
No real users.  
No actual burgers were harmed.

<br>
