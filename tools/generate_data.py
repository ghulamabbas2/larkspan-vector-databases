"""Generate the Larkspan dataset (customers + support tickets).

Larkspan is a fictional B2B company that sells a project-management and
helpdesk product. Everything here is made up. The generator is seeded, so it
produces exactly the same files every time:

    uv run python tools/generate_data.py

Writes:
    data/larkspan/customers.json        50 customer companies (tenants)
    data/larkspan/tickets.jsonl         5,000 support tickets
    data/larkspan/error_codes.json      what each ERR-xxxx code means
    eval/ticket_topics.json             hidden ground truth: the topic of every ticket
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
N_CUSTOMERS = 50
N_TICKETS = 5000
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "larkspan"
EVAL = ROOT / "eval"

# ── Customers ────────────────────────────────────────────────────────────────
FIRST = ["Harbor", "Juniper", "Copperfield", "Bluewater", "Northgate", "Saltmarsh", "Redcliff", "Mossgrove",
         "Ironbridge", "Willowbank", "Stonehaven", "Clearbrook", "Ashford", "Brightwater", "Eastvale",
         "Oakhurst", "Riverside", "Silverline", "Thornbury", "Westmere", "Kestrel", "Marlow", "Pinecrest",
         "Larchmont", "Fairhollow"]
SECOND = ["Dental Group", "Logistics", "Architects", "Health", "Media", "Legal", "Engineering", "Foods",
          "Learning", "Property", "Studios", "Analytics", "Outfitters", "Consulting", "Labs"]
PLANS = [("Starter", 0.4), ("Growth", 0.4), ("Enterprise", 0.2)]
REGIONS = ["us-east", "us-west", "eu-west", "eu-central", "ap-southeast"]

# ── Error codes (each has one exact meaning, like a real product) ────────────
ERROR_CODES = {
    "ERR-4012": "Webhook signature verification failed (the signing secret doesn't match).",
    "ERR-4031": "SSO assertion expired (usually the identity provider's clock is out of sync).",
    "ERR-4104": "Automation loop detected (an automation triggered itself too many times).",
    "ERR-4220": "Custom field type mismatch (a value doesn't fit the field's type).",
    "ERR-4290": "API rate limit exceeded.",
    "ERR-4403": "Guest users can't access this board.",
    "ERR-5007": "Import file is too large.",
    "ERR-5031": "Report export timed out.",
    "ERR-5102": "Mobile sync conflict (the same task was edited offline on two devices).",
}

# ── Slot values ──────────────────────────────────────────────────────────────
SLOTS = {
    "project": ["Q3 Launch", "Website Redesign", "Onboarding 2026", "Warehouse Move", "Spring Campaign",
                "Client Portal", "Office Fit-out", "Data Migration", "Audit Prep", "Mobile App v2",
                "Hiring Plan", "Product Roadmap", "Support Backlog", "Brand Refresh"],
    "browser": ["Chrome", "Safari", "Firefox", "Edge"],
    "device": ["iPhone 15", "iPhone 16", "Pixel 9", "Galaxy S24", "iPad Air", "an Android tablet"],
    "integration": ["Slack", "GitHub", "Google Drive", "Microsoft Teams", "Dropbox", "Google Calendar"],
    "idp": ["Okta", "Microsoft Entra ID", "Google Workspace", "OneLogin"],
    "n": ["3", "5", "8", "12", "20", "40"],
    "big": ["15,000", "40,000", "120,000", "250,000"],
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "role": ["Admin", "Member", "Guest"],
    "time": ["this morning", "yesterday", "since the weekend", "for two days", "since last week", "since the last update"],
    "teammate": ["my manager", "a new hire", "our contractor", "two people on my team", "our finance lead", "a client"],
    "fmt": ["CSV", "Excel", "PDF"],
    "card": ["Visa", "Mastercard", "Amex"],
    "view": ["board", "timeline", "calendar", "list"],
}

# ── Ticket topics ────────────────────────────────────────────────────────────
# Each topic: product area, a few subject lines, a few bodies, optional error code,
# and the resolution an agent wrote. Many phrasings per topic, so the same problem
# is described with different words (that's what semantic search has to handle).
TOPICS = [
    # Account & Login
    dict(id="login-password-reset-expired", area="Account & Login",
         subjects=["Password reset link expired", "Reset email link doesn't work", "Can't reset my password",
                   "Reset link says it is invalid"],
         bodies=["I asked for a password reset {time} but the link says it has expired as soon as I click it.",
                 "Every time I open the reset email and click the button I get 'this link is no longer valid'. Tried {n} times.",
                 "I forgot my password and the reset link doesn't work. I'm stuck on the sign-in page and can't get to my projects."],
         resolution="Reset links expire after 30 minutes and only the newest link works. Asked the customer to request one new link and use it right away."),
    dict(id="login-2fa-lost-phone", area="Account & Login",
         subjects=["Lost my phone, can't get past 2FA", "Locked out by two-factor authentication",
                   "New phone, authenticator codes don't work", "2FA code rejected"],
         bodies=["I got a new phone and my authenticator app didn't transfer. Now I can't sign in at all.",
                 "Two-factor keeps rejecting my code and I don't have the backup codes. I'm completely locked out of my account.",
                 "My phone was stolen {time}. How do I get back into Larkspan without the 2FA codes?"],
         resolution="Verified identity with the workspace admin, reset 2FA on the account, and reminded the user to save backup codes."),
    dict(id="login-sso-assertion-expired", area="Account & Login", code="ERR-4031",
         subjects=["SSO login fails with ERR-4031", "Single sign-on stopped working", "{idp} login error",
                   "Team can't sign in through {idp}"],
         bodies=["Since {time} everyone who signs in with {idp} gets ERR-4031. Password login still works for the owner account.",
                 "Our SSO broke. People click 'Sign in with {idp}', get redirected back, and see an error about an expired assertion.",
                 "ERR-4031 on sign in. Nobody on the team can get in.",
                 "Single sign-on is failing for about half of our users. Others are fine. We use {idp}."],
         resolution="The identity provider's server clock was several minutes off, so assertions arrived already expired. Customer synced the clock (NTP) and SSO worked again."),
    dict(id="login-account-locked", area="Account & Login",
         subjects=["Account locked after failed attempts", "Too many login attempts", "Locked out of my account",
                   "Says my account is temporarily locked"],
         bodies=["I typed my password wrong a few times and now it says my account is locked. How long does that last?",
                 "{teammate} can't get in. The sign-in page says 'too many attempts, try again later'.",
                 "I can't get into my account at all. It says it's locked for security reasons."],
         resolution="Accounts unlock automatically after 15 minutes. Unlocked it manually and suggested a password reset."),
    dict(id="login-email-change", area="Account & Login",
         subjects=["How do I change my login email?", "Need to update my email address", "Changed company domain, update emails"],
         bodies=["Our company moved to a new domain. How do we change everyone's sign-in email without losing their history?",
                 "I want to change the email I use to sign in. I can't find the setting.",
                 "My email address changed after a name change. Can you update it on my account?"],
         resolution="Email can be changed under Profile > Account. For a domain change, admins can bulk-update emails under Admin > Members."),

    # Billing
    dict(id="billing-double-charge", area="Billing",
         subjects=["Charged twice this month", "Duplicate charge on our card", "Two invoices for the same period",
                   "We were billed twice"],
         bodies=["Our {card} was charged twice for this month's subscription. Please refund the duplicate.",
                 "I see two identical payments on our statement for Larkspan. Only one should be there.",
                 "Finance noticed a double charge. Can you check and reverse one of them?"],
         resolution="The first payment failed and was retried, but the bank showed both as pending. The extra pending charge drops off within 5 to 7 days. No refund needed."),
    dict(id="billing-refund-request", area="Billing",
         subjects=["Refund request", "Can we get a refund?", "Cancel and refund annual plan",
                   "Want money back for unused months"],
         bodies=["We bought the annual plan {time} but the project was cancelled. Can we get a refund for the unused months?",
                 "I upgraded by mistake and want my money back.",
                 "We'd like to cancel and get a refund. We haven't really used the product."],
         resolution="Explained the refund policy: full refund within 14 days, after that a prorated credit for annual plans. Processed per policy."),
    dict(id="billing-invoice-vat", area="Billing",
         subjects=["Need VAT number on invoice", "Invoice missing company details", "Change billing address on invoices"],
         bodies=["Our accountant needs our VAT ID and company address on the invoices. Where do I add them?",
                 "The invoices only show my name, not the company. Can you fix past invoices too?",
                 "We need to change the billing address that appears on our invoices."],
         resolution="Billing details are under Admin > Billing > Invoice details. Regenerated the last 3 invoices with the new details."),
    dict(id="billing-seat-count", area="Billing",
         subjects=["Billed for more seats than we use", "Why are we paying for {n} extra seats?", "Seat count wrong on invoice"],
         bodies=["We removed {n} people last month but the invoice still charges for them.",
                 "Our bill went up and I don't understand why. We didn't add anyone.",
                 "The number of seats on the invoice doesn't match the number of active members."],
         resolution="Deactivated members still held paid seats until removed under Admin > Billing > Seats. Removed them and credited the difference."),
    dict(id="billing-card-declined", area="Billing",
         subjects=["Card declined", "Payment failed", "Can't update credit card", "Subscription payment didn't go through"],
         bodies=["We got an email that our payment failed. The card is valid, we use it everywhere else.",
                 "I'm trying to add a new {card} but it keeps saying the card was declined.",
                 "Our renewal payment failed and now there's a banner saying the workspace will be downgraded."],
         resolution="The bank blocked the charge as a foreign transaction. Customer approved it with the bank and the retry succeeded."),

    # Projects & Tasks
    dict(id="tasks-deleted-recover", area="Projects & Tasks",
         subjects=["Accidentally deleted a project", "Recover deleted tasks", "Project disappeared",
                   "Undo delete on {project}"],
         bodies=["{teammate} deleted the {project} project by mistake. Is there any way to get it back?",
                 "A whole list of tasks vanished from {project}. We think someone deleted them.",
                 "I deleted a task with a lot of comments on it. Can it be restored?"],
         resolution="Deleted items stay in Trash for 30 days. Restored the project from Admin > Trash."),
    dict(id="tasks-recurring-not-created", area="Projects & Tasks",
         subjects=["Recurring task not created", "Repeating tasks stopped", "Weekly task didn't show up"],
         bodies=["We have a weekly task that should appear every {weekday}. It didn't show up this week.",
                 "Our recurring tasks stopped generating {time}.",
                 "The repeat setting on our monthly report task doesn't seem to do anything anymore."],
         resolution="The recurrence was tied to a task that had been archived, which stops the series. Unarchived it and the next task was created."),
    dict(id="tasks-dependencies-dates", area="Projects & Tasks",
         subjects=["Dependencies don't move dates", "Changing one date doesn't update the others",
                   "Task dependencies not working"],
         bodies=["When I move a task later, the tasks that depend on it don't shift. I thought that was automatic.",
                 "We set up dependencies in {project} but the due dates don't update when the first task slips.",
                 "Is there a way to make follow-up tasks move when an earlier task is delayed?"],
         resolution="Auto-scheduling is off by default. Turned it on under Project settings > Dependencies."),
    dict(id="tasks-custom-field-mismatch", area="Projects & Tasks", code="ERR-4220",
         subjects=["ERR-4220 when saving a task", "Can't save custom field value", "Custom field error"],
         bodies=["Getting ERR-4220 whenever I try to save a budget amount on a task.",
                 "The custom field for cost won't accept '1,200 USD'. It throws an error.",
                 "ERR-4220 on save in {project}.",
                 "Our date field rejects dates typed as 03/04/2026 and shows an error."],
         resolution="The field is a Number type, so it rejects text like 'USD' and thousands separators. Explained the accepted format."),
    dict(id="tasks-bulk-edit", area="Projects & Tasks",
         subjects=["How to change many tasks at once?", "Bulk edit tasks", "Reassign all tasks from one person"],
         bodies=["Someone left the company and I need to move all {n} of their tasks to a new owner.",
                 "Is there a way to change the due date on a lot of tasks at the same time?",
                 "I need to tag about {n} tasks with the same label. Doing it one by one is painful."],
         resolution="Showed multi-select (Shift+click) and the bulk action bar, which can reassign, re-date and tag."),

    # Boards & Views
    dict(id="views-board-slow", area="Performance",
         subjects=["Board takes forever to load", "Larkspan is very slow", "Pages load slowly",
                   "Timeline view freezes"],
         bodies=["Our main board has about {big} tasks and it takes almost a minute to open in {browser}.",
                 "Everything is sluggish {time}. Clicking a task takes several seconds.",
                 "The {view} view freezes the browser tab when I scroll."],
         resolution="The board showed all archived tasks. Turned on 'Hide archived' and added a filter; load time dropped to about 2 seconds."),
    dict(id="views-filter-saved-lost", area="Boards & Views",
         subjects=["Saved filter disappeared", "My saved views are gone", "Filters reset every time"],
         bodies=["I had a saved filter for my tasks and it's gone. Did something change?",
                 "Each time I open the {view} view my filters are reset to the default.",
                 "Our team's shared views vanished from the sidebar."],
         resolution="Personal views were hidden by a sidebar setting after the update. Restored them via View menu > Show saved views."),
    dict(id="views-timeline-missing-tasks", area="Boards & Views",
         subjects=["Tasks missing from timeline", "Timeline doesn't show all tasks", "Gantt chart empty"],
         bodies=["Half of the tasks in {project} don't appear on the timeline.",
                 "The timeline view is empty even though the project has plenty of tasks.",
                 "Some tasks show on the board but not in the Gantt view."],
         resolution="Tasks without both a start and a due date are not drawn on the timeline. Explained and showed the 'Show unscheduled' panel."),

    # Automations
    dict(id="automation-loop", area="Automations", code="ERR-4104",
         subjects=["Automation stopped with ERR-4104", "Automation disabled itself", "Rule keeps turning off"],
         bodies=["Our automation that moves tasks to Done was switched off with ERR-4104.",
                 "One of our rules keeps disabling itself and we get an email about a loop.",
                 "ERR-4104 again. Third time this week.",
                 "An automation updated the same tasks over and over, then stopped working."],
         resolution="The rule's action (change status) also matched its own trigger (status changed), so it kept firing itself. Added a condition to break the loop."),
    dict(id="automation-not-firing", area="Automations",
         subjects=["Automation not triggering", "Rule doesn't run", "When a task moves, nothing happens"],
         bodies=["We set up a rule to notify the client when a task is marked Done. It never fires.",
                 "Our automation worked last month but {time} it doesn't run at all.",
                 "The 'assign reviewer' rule only works sometimes."],
         resolution="The rule was limited to one project and the tasks lived in another. Changed the rule's scope."),
    dict(id="automation-limit", area="Automations",
         subjects=["Hit the automation limit", "Monthly automation runs used up", "Automations paused for the month"],
         bodies=["We got a message that we've used all automation runs for the month. What are our options?",
                 "Our automations are paused and it says we reached the plan limit.",
                 "How many automation runs does the Growth plan include?"],
         resolution="Starter includes 250 runs per month, Growth 5,000, Enterprise unlimited. Offered an add-on pack for the rest of the month."),

    # Integrations
    dict(id="integration-slack-no-messages", area="Integrations",
         subjects=["Slack notifications stopped", "No messages in Slack", "Slack integration broken"],
         bodies=["We used to get a Slack message whenever a task was assigned. Nothing {time}.",
                 "The Slack integration shows as connected but no messages arrive in our channel.",
                 "Updates from {project} aren't posting to Slack anymore."],
         resolution="The Slack channel was renamed and made private, so the Larkspan app lost access. Re-invited the app to the channel."),
    dict(id="integration-github-link", area="Integrations",
         subjects=["GitHub pull requests don't link to tasks", "GitHub integration not updating tasks",
                   "Commits not showing on tasks"],
         bodies=["When we mention the task ID in a pull request, nothing appears on the task.",
                 "Our GitHub integration stopped moving tasks to Done when PRs are merged.",
                 "We connected GitHub but commits never show up in the activity feed."],
         resolution="The GitHub app was only installed on some repositories. Added the missing repositories in GitHub's app settings."),
    dict(id="integration-drive-permission", area="Integrations",
         subjects=["Can't attach Google Drive files", "Drive files show 'access denied'", "Google Drive attachments broken"],
         bodies=["When I attach a Google Drive file, my teammates see 'You need access'.",
                 "Google Drive files attached to tasks can't be opened by anyone but me.",
                 "Drive integration asks me to sign in again every day."],
         resolution="Attaching a link doesn't change Drive sharing. Explained that the file must be shared in Drive, or use the 'share with project' option."),
    dict(id="integration-calendar-sync", area="Integrations",
         subjects=["Due dates not in my calendar", "Calendar sync stopped", "Google Calendar shows old dates"],
         bodies=["My task due dates used to show in Google Calendar. Now changes don't sync.",
                 "The calendar feed still shows tasks that were completed weeks ago.",
                 "We moved deadlines in {project} but Outlook still has the old ones."],
         resolution="Calendar feeds refresh on the calendar provider's schedule, which can take up to 24 hours. Suggested re-subscribing to force a refresh."),

    # API & Webhooks
    dict(id="webhook-signature", area="API & Webhooks", code="ERR-4012",
         subjects=["Webhook deliveries failing with ERR-4012", "Webhooks rejected", "Webhook signature error",
                   "ERR-4012 on our endpoint"],
         bodies=["Every webhook delivery to our endpoint fails with ERR-4012 {time}.",
                 "Our server rejects Larkspan webhooks because the signature doesn't match.",
                 "Seeing ERR-4012 again. Can someone look?",
                 "We rotated secrets last week and now webhook verification fails on our side."],
         resolution="The customer rotated the signing secret in Larkspan but kept the old one on their server. Updated the secret on their side; deliveries succeeded."),
    dict(id="api-rate-limit", area="API & Webhooks", code="ERR-4290",
         subjects=["API returns ERR-4290", "Rate limited by the API", "Too many requests error"],
         bodies=["Our sync script gets ERR-4290 after about a minute of running.",
                 "We're hitting rate limits when importing tasks through the API.",
                 "ERR-4290 in production since we added more users.",
                 "The API keeps answering 429 and our nightly job fails."],
         resolution="The limit is 100 requests per minute per token. Recommended batching endpoints and backing off using the Retry-After header."),
    dict(id="api-token-revoked", area="API & Webhooks",
         subjects=["API token stopped working", "401 unauthorized from API", "Personal access token invalid"],
         bodies=["Our integration suddenly gets 401 Unauthorized on every call.",
                 "The API token we use for reporting stopped working {time}.",
                 "Did tokens expire? Our scripts all fail authentication."],
         resolution="The token belonged to a user who was deactivated, which revokes their tokens. Created a service account token instead."),

    # Notifications & Email
    dict(id="notifications-too-many", area="Notifications & Email",
         subjects=["Too many email notifications", "How do I stop all these emails?", "Notification overload"],
         bodies=["I get an email for every single comment. It's hundreds a day.",
                 "Our team is drowning in notifications. Can we turn some off for everyone?",
                 "How do I only get emails when I'm mentioned?"],
         resolution="Showed Notification settings: switch to 'Mentions and assignments only' and use the daily digest."),
    dict(id="notifications-not-received", area="Notifications & Email",
         subjects=["Not receiving any emails", "Email notifications missing", "Invite email never arrived"],
         bodies=["{teammate} never received the invite email.",
                 "I stopped getting notification emails {time}. Nothing in spam either.",
                 "Our whole company doesn't receive Larkspan emails."],
         resolution="The company mail filter was blocking our sending domain. Shared the domains to allow-list."),

    # Reports & Dashboards
    dict(id="report-export-timeout", area="Reports & Dashboards", code="ERR-5031",
         subjects=["Report export fails with ERR-5031", "Export times out", "Can't download report as {fmt}"],
         bodies=["Exporting our yearly time report to {fmt} fails with ERR-5031.",
                 "The export spinner runs for ages and then says it timed out.",
                 "ERR-5031 when I export.",
                 "Large reports never finish downloading."],
         resolution="Very large exports time out in the browser. Suggested narrowing the date range or using scheduled exports, which are emailed when ready."),
    dict(id="report-numbers-wrong", area="Reports & Dashboards",
         subjects=["Dashboard numbers look wrong", "Report totals don't match", "Time tracking totals incorrect"],
         bodies=["The hours on the dashboard don't match the hours in the task list.",
                 "Our completed-tasks chart shows fewer tasks than we actually closed.",
                 "Report totals changed overnight without anyone editing anything."],
         resolution="The dashboard uses the workspace time zone, while the list used the user's local time zone. Aligned the time zone setting."),

    # Mobile
    dict(id="mobile-sync-conflict", area="Mobile App", code="ERR-5102",
         subjects=["Mobile app shows ERR-5102", "Sync conflict on mobile", "Changes lost after going offline"],
         bodies=["I edited tasks on the plane and when I landed the app showed ERR-5102.",
                 "My changes from the mobile app disappeared after syncing.",
                 "ERR-5102 on my {device}.",
                 "Two of us edited the same task offline and now one version is gone."],
         resolution="Both devices edited the same task offline; the later sync wins. Showed the task history where the overwritten version can be restored."),
    dict(id="mobile-push-missing", area="Mobile App",
         subjects=["No push notifications on phone", "Mobile notifications not working", "App doesn't alert me"],
         bodies=["I don't get push notifications on my {device} anymore.",
                 "The mobile app never notifies me about new assignments.",
                 "Notifications work on desktop but not on the phone."],
         resolution="Notifications were disabled at the phone's system level after an OS update. Re-enabled them in the phone settings."),
    dict(id="mobile-crash", area="Mobile App",
         subjects=["App crashes on startup", "Mobile app keeps closing", "App freezes when opening a task"],
         bodies=["The app crashes as soon as I open it on my {device}.",
                 "Every time I open a task with images, the app closes.",
                 "The mobile app freezes {time}."],
         resolution="Known issue in the previous version with large image attachments. Fixed in the latest app update."),

    # Import / Export
    dict(id="import-file-too-large", area="Import & Export", code="ERR-5007",
         subjects=["Import fails with ERR-5007", "CSV import too big", "Can't import our spreadsheet"],
         bodies=["Trying to import a spreadsheet with {big} rows and it fails with ERR-5007.",
                 "Our migration file is rejected as too large.",
                 "ERR-5007 on import.",
                 "The importer won't accept our export from our old tool."],
         resolution="Imports are limited to 50 MB or 100,000 rows per file. Split the file into parts."),
    dict(id="import-dates-wrong", area="Import & Export",
         subjects=["Dates wrong after import", "Imported due dates are shifted", "Day and month swapped on import"],
         bodies=["After importing our tasks, all the due dates are a month off.",
                 "Dates like 04/05 were imported as April 5th but we meant May 4th.",
                 "Every imported date moved one day earlier."],
         resolution="The importer assumed US date format. Re-ran the import with the date format set to DD/MM/YYYY."),

    # Permissions
    dict(id="permissions-guest-board", area="Permissions & Roles", code="ERR-4403",
         subjects=["Client sees ERR-4403", "Guest can't open the board", "Shared board shows access error"],
         bodies=["We invited a client as a guest but they get ERR-4403 when they open the board.",
                 "Our contractor can't see the project we shared.",
                 "ERR-4403 for every guest.",
                 "A guest user says the link shows 'access denied'."],
         resolution="Guests only see items shared with them directly. Shared the board with the guest instead of the whole project."),
    dict(id="permissions-member-sees-too-much", area="Permissions & Roles",
         subjects=["Members can see private projects", "How to hide a project from some people?",
                   "Restrict access to HR project"],
         bodies=["We have a confidential HR project and I need only {n} people to see it.",
                 "Someone in sales could see our finance board. How do we lock it down?",
                 "Can a {role} see projects they're not invited to?"],
         resolution="Set the project to Private and added only the people who need it. Explained role permissions."),
    dict(id="permissions-transfer-owner", area="Permissions & Roles",
         subjects=["Transfer workspace ownership", "Owner left the company", "Change the account owner"],
         bodies=["The person who created our workspace has left. How do we make someone else the owner?",
                 "I need to transfer ownership of the workspace to our new IT lead.",
                 "Our owner account belongs to a former employee and we can't change billing."],
         resolution="Ownership can be transferred by the current owner, or by support after verification from another admin. Completed the transfer."),

    # Helpdesk product
    dict(id="helpdesk-sla-breach", area="Helpdesk & SLA",
         subjects=["SLA timer wrong", "SLA breached even though we replied", "SLA ignores business hours"],
         bodies=["Tickets show as SLA breached even though we answered within the hour.",
                 "The SLA clock keeps running over the weekend.",
                 "Our first-response SLA counts internal notes as replies."],
         resolution="The SLA policy used 24/7 hours. Switched it to the business-hours calendar."),
    dict(id="helpdesk-email-forwarding", area="Helpdesk & SLA",
         subjects=["Support emails not creating tickets", "Forwarded emails don't arrive", "Inbox not receiving mail"],
         bodies=["We forward support@ to Larkspan but no tickets are created.",
                 "Customer emails stopped turning into tickets {time}.",
                 "Some emails arrive as tickets, others never show up."],
         resolution="The forwarding rule was sending to an old inbox address. Updated it to the address shown in Helpdesk settings."),
    dict(id="helpdesk-macros", area="Helpdesk & SLA",
         subjects=["How to create canned replies?", "Saved responses for common questions", "Macros for agents"],
         bodies=["Our agents type the same answers all day. Is there a way to save replies?",
                 "Can a macro also change the ticket status?",
                 "How do I share saved replies with the whole team?"],
         resolution="Showed Macros: saved replies that can also set status, tags and assignee, shared at team level."),

    # Account admin / data
    dict(id="admin-data-export-gdpr", area="Security & Privacy",
         subjects=["Need all our data for a GDPR request", "Delete a user's personal data", "Data export for compliance"],
         bodies=["A former employee asked for all their personal data under GDPR. How do we export it?",
                 "We need to permanently delete one person's data from the workspace.",
                 "Our auditors want a full export of the workspace."],
         resolution="Admins can export a member's data and request deletion under Admin > Privacy. Full workspace exports are available on Enterprise."),
    dict(id="security-suspicious-login", area="Security & Privacy",
         subjects=["Suspicious login alert", "Someone signed in from another country", "Unknown device on my account"],
         bodies=["I got an email about a new sign-in from a country I've never been to.",
                 "There's a device I don't recognise in my active sessions.",
                 "Was my account hacked? I got a login alert at 3am."],
         resolution="Signed out all sessions, reset the password and turned on 2FA. The sign-in came from a VPN the user had forgotten about."),
]

OPENERS = ["", "", "", "Hi team, ", "Hello, ", "Hi, ", "Urgent: ", "Quick question. ", "Hey support, "]
CLOSERS = ["", "", "", " Thanks!", " Thanks in advance.", " Please help.", " Any ideas?", " This is blocking us.",
           " Can you check?", " Appreciate it."]
PRIORITIES = [("low", 0.2), ("normal", 0.5), ("high", 0.22), ("urgent", 0.08)]
STATUSES = [("closed", 0.55), ("solved", 0.25), ("pending", 0.1), ("open", 0.1)]
CHANNELS = [("email", 0.5), ("web", 0.3), ("chat", 0.2)]


def pick(rng, weighted):
    return rng.choices([v for v, _ in weighted], weights=[w for _, w in weighted])[0]


def fill(rng, text):
    for key, values in SLOTS.items():
        token = "{" + key + "}"
        while token in text:
            text = text.replace(token, rng.choice(values), 1)
    return text


def make_customers(rng):
    names = set()
    while len(names) < N_CUSTOMERS:
        names.add(f"{rng.choice(FIRST)} {rng.choice(SECOND)}")
    customers = []
    for i, name in enumerate(sorted(names), start=1):
        plan = pick(rng, PLANS)
        seats = {"Starter": rng.randint(3, 15), "Growth": rng.randint(15, 120), "Enterprise": rng.randint(120, 2000)}[plan]
        customers.append({
            "id": f"cus_{i:03d}", "name": name, "plan": plan, "seats": seats,
            "region": rng.choice(REGIONS),
            "customer_since": (datetime(2021, 1, 1) + timedelta(days=rng.randint(0, 1400))).date().isoformat(),
        })
    return customers


def make_tickets(rng, customers):
    start = datetime(2025, 1, 1)
    span = int((datetime(2026, 9, 30) - start).total_seconds())
    # bigger customers file more tickets
    weights = [c["seats"] ** 0.5 for c in customers]
    tickets, truth = [], {}
    seen = set()
    while len(tickets) < N_TICKETS:
        topic = rng.choice(TOPICS)
        subject = fill(rng, rng.choice(topic["subjects"]))
        body = fill(rng, rng.choice(OPENERS) + rng.choice(topic["bodies"]) + rng.choice(CLOSERS))
        if (subject, body) in seen and rng.random() < 0.7:
            continue  # keep exact duplicates rare (real helpdesks do have some)
        seen.add((subject, body))
        cust = rng.choices(customers, weights=weights)[0]
        status = pick(rng, STATUSES)
        tid = f"TCK-{10001 + len(tickets)}"
        tickets.append({
            "id": tid,
            "customer_id": cust["id"],
            "plan": cust["plan"],
            "product_area": topic["area"],
            "priority": pick(rng, PRIORITIES),
            "status": status,
            "channel": pick(rng, CHANNELS),
            "created_at": (start + timedelta(seconds=rng.randint(0, span))).isoformat(timespec="seconds"),
            "subject": subject,
            "body": body,
            "resolution": topic["resolution"] if status in ("solved", "closed") else None,
        })
        truth[tid] = topic["id"]
    tickets.sort(key=lambda t: t["created_at"])
    # renumber in date order so IDs grow with time, like a real helpdesk
    renamed, new_truth = [], {}
    for i, t in enumerate(tickets):
        new_id = f"TCK-{10001 + i}"
        new_truth[new_id] = truth[t["id"]]
        renamed.append({**t, "id": new_id})
    return renamed, new_truth


def main():
    rng = random.Random(SEED)
    customers = make_customers(rng)
    tickets, truth = make_tickets(rng, customers)
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    (OUT / "customers.json").write_text(json.dumps(customers, indent=2) + "\n")
    with (OUT / "tickets.jsonl").open("w") as f:
        for t in tickets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    (OUT / "error_codes.json").write_text(json.dumps(ERROR_CODES, indent=2) + "\n")
    topics = {t["id"]: {"area": t["area"], "code": t.get("code")} for t in TOPICS}
    (EVAL / "ticket_topics.json").write_text(json.dumps({"topics": topics, "tickets": truth}, indent=1) + "\n")
    print(f"{len(customers)} customers, {len(tickets)} tickets, {len(TOPICS)} topics → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
