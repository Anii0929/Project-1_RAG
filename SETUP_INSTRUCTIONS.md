# Setup Instructions

## Error Fix: "Query failed" with 500 Internal Server Error

The error you're seeing is likely due to one of these API issues. Here's how to fix them:

### 1. Create a .env file

Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Add your Anthropic API key

Edit the `.env` file and replace `your-anthropic-api-key-here` with your actual Anthropic API key:

```bash
# .env file content
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-api-key-here
```

### 3. Get an Anthropic API key

If you don't have an Anthropic API key:
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API keys section
4. Create a new API key
5. Copy the key to your `.env` file

### 4. Add credits to your Anthropic account

**Important**: Even with a valid API key, you need credits in your Anthropic account:
1. Go to https://console.anthropic.com/settings/billing
2. Add credits or upgrade your plan
3. The API requires a minimum credit balance to function

### 5. Restart the application

After adding the API key and ensuring you have credits, restart the server:
```bash
./run.sh
```

## Common Error Messages

- **"API credit balance is too low"**: Add credits to your Anthropic account
- **"Invalid API key"**: Check your ANTHROPIC_API_KEY in the .env file
- **"Query failed"**: Usually indicates insufficient credits or invalid API key

## Verification

Once the API key is set up correctly, you should be able to:
1. Ask questions about course materials
2. See responses with clickable source citations
3. Click on sources to open lesson videos in new tabs

## Clickable Links Feature

The application now supports clickable lesson links! When you ask questions about course content:
- Source citations will appear as clickable links
- Clicking a source opens the corresponding lesson video in a new tab
- Links are retrieved automatically from the course documents