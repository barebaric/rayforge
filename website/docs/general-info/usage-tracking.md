# Usage Tracking

Rayforge includes optional anonymous usage tracking to help us understand how the application is used and prioritize future development. This page explains what we track, how it works, and your privacy.

## Opt-In Only

Usage tracking is **completely optional**. When you first start Rayforge, you'll be asked whether you want to participate:

- **Yes**: Anonymous usage data will be sent to our analytics server
- **No**: No data is ever collected or transmitted

You can change this choice at any time in the General settings.

## What We Track

When enabled, we collect only anonymous page view data - similar to website analytics. Here's what we can see:

| Data                 | Example                   |
| -------------------- | ------------------------- |
| Screen resolution    | 1920x1080                 |
| Language setting     | en-US                     |
| Pages/dialogs viewed | /machine-settings/general |
| Time spent on page   | 6m 3s                     |

## What We See

Here's an example of what the analytics dashboard looks like:

| Path                      | Visitors | Visits | Views | Bounce rate | Visit duration |
| ------------------------- | -------- | ------ | ----- | ----------- | -------------- |
| /                         | 1        | 1      | 5     | 0%          | 27m 35s        |
| /machine-settings/general | 1        | 1      | 5     | 0%          | 27m 27s        |
| /view/3d                  | 1        | 1      | 2     | 0%          | 25m 14s        |
| /camera-alignment-dialog  | 1        | 1      | 2     | 0%          | 6m 3s          |
| /machine-settings/camera  | 1        | 1      | 2     | 0%          | 6m 16s         |
| /settings/general         | 1        | 1      | 2     | 0%          | 16m 36s        |
| /step-settings/rasterizer | 1        | 1      | 2     | 0%          | 11s            |

## What We Do NOT Track

We are committed to your privacy:

- **No personal information** - No names, emails, or accounts
- **No file contents** - Your designs and projects stay private
- **No machine identifiers** - No serial numbers or unique IDs
- **No IP addresses stored** - We use Umami analytics which doesn't store IPs
- **No cross-site tracking** - Data is isolated to Rayforge only

## Why We Track

Usage data helps us:

- **Identify popular features** - Know what's working well
- **Find pain points** - See where users spend time or get stuck
- **Prioritize development** - Focus on features people actually use
- **Understand diversity** - Know what languages and screen sizes to support

## How It Works

Rayforge uses [Umami](https://umami.is/), an open-source, privacy-focused analytics platform. The tracking:

- Sends small HTTP requests in the background
- Does not affect application performance
- Works offline (failed requests are silently ignored)
- Uses a generic User-Agent to prevent fingerprinting

## Disabling Tracking

You can disable tracking at any time:

1. Open **Settings** → **General**
2. Toggle off **Send anonymous usage statistics**

When disabled, absolutely no data is sent.

## Related Pages

- **[Application Settings](../ui/settings)** - Configure tracking preferences
