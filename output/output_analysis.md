## What the Results Tell You:
Looking at the output, the **false positives** are being flagged mainly because of:

* `login_success_encoded` (failed logins are suspicious)
* `device_encoded` (tablets are rare in your data - only 5%)
* `sessions_per_hour` = 0 (unusual to have zero sessions)

The **true positive** (China login) is flagged because of:

* `sessions_per_hour` = 20 (way higher than normal)
* `hour_of_day` = 2am (low traffic time)
* `country` = CN (not in your normal training data)

This makes sense! The model is working, just being overly cautious about rare combinations.
