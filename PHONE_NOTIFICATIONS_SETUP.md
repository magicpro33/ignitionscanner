# IGNITION — Phone Notification Setup (ntfy.sh)

Get IGNITION's alerts pushed to your phone in about 5 minutes. Free, no
account required.

---

## How it works

The scanner sends alerts to a **topic** on ntfy.sh — think of a topic as a
private channel. Your phone subscribes to that topic and gets an instant
push whenever the scanner posts to it.

**The topic name is the only password.** Anyone who knows it can read your
alerts. Make it unguessable and keep it private.

---

## Step 1 — Pick your topic name

Make up a long, random topic name. Good and bad examples:

| Good | Bad |
|---|---|
| `brad-ignition-x7k2m9qv` | `brad-stocks` |
| `ign-scanner-4hq8w2zt9` | `ignition` |

Never commit it to GitHub or post it anywhere public.

## Step 2 — Set up your phone

1. Install the **ntfy** app — free on the App Store (iOS) or Google Play
   (Android).
2. Open the app and tap **+** (Subscribe to topic).
3. Type your exact topic name from Step 1 and confirm.
4. Recommended: open the topic's settings in the app and allow alerts to
   bypass Do Not Disturb. IGNITING alerts are sent at **urgent** priority
   specifically so they can punch through during market hours if you
   allow it.

## Step 3 — Tell the scanner your topic

Add one line to the same secrets file that holds your Alpaca keys:

```toml
NTFY_TOPIC = "brad-ignition-x7k2m9qv"
```

**Running locally:** put it in `.streamlit/secrets.toml` in the app folder
(create the `.streamlit` folder if it doesn't exist).

**Running on Streamlit Cloud:** go to share.streamlit.io, open your app's
three-dot menu, then **Settings -> Secrets**, and paste the line into the
box. It takes effect on the next rerun. Do NOT put the secrets file in
the GitHub repo.

If you ever self-host an ntfy server, add `NTFY_SERVER = "https://your-server"`
too. Otherwise leave it out — the default is ntfy.sh.

## Step 4 — Test it

1. Restart / rerun the app.
2. The sidebar now shows a **Phone notifications (ntfy)** toggle.
3. Tap **Send test notification**.
4. Your phone should buzz within a second or two. If it does, you're done.

---

## What the alerts mean

| Push | Priority | Meaning |
|---|---|---|
| `IGNITING: SMR @ $24.18` | Urgent | All four live conditions confirmed at once on a flat/up day — the footprint of a fresh momentum leg. |
| `GAP REVERSAL: OXM @ $35.87` | High | Same footprint, but the stock is down 4%+ on the day or gapped down 4%+. A bounce attempt inside a selloff — tradable, but a different and riskier trade. |
| `DBI alert @ $6.56` | Default | The ticker's overall Score crossed your alert threshold (sidebar slider). |

Each ticker alerts at most **once per day**, so a volatile name won't spam
you every refresh.

---

## Important limitation

Pushes only fire while the scanner is actually running — that means a
browser tab with the app open somewhere (minimized is fine). Close every
tab and the scan loop stops, and so do the alerts. For alerts with zero
tabs open, run a headless watcher version of the scan loop on an
always-on machine.

## Troubleshooting

- **Test button says "Send failed"** — check the topic name in secrets for
  typos or stray spaces; confirm the app has internet access.
- **Test succeeds but the phone is silent** — make sure the phone's ntfy
  app is subscribed to the *exact* same topic string (it's
  case-sensitive), and check the app's notification permissions in your
  phone settings.
- **Alerts arrive late on iPhone** — iOS can delay background pushes when
  Low Power Mode is on; the ntfy app's "instant delivery" option helps on
  Android, and on iOS keeping the app recently opened helps.
- **No alerts all day** — that can be correct behavior. If nothing crossed
  your threshold and nothing ignited, the scanner stays quiet. Lower the
  Alert score threshold slider if you want a chattier feed.

---

*Not financial advice. The scanner detects momentum early; it does not
predict the future.*
