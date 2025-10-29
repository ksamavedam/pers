#!/usr/bin/env python3
"""
Daily Market Analysis Script
Runs Claude analysis and emails results
"""

import anthropic
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

# Configuration

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = "ksamavedam@gmail.com"  # Can be same or different

# Your market analysis prompt
MARKET_PROMPT = """
You are a market analyst providing daily pre-market sector and ETF recommendations. 
Current date: {date}
Market open time: 9:30 AM ET

## Your Task
Analyze current market conditions and recommend:
1. Top 3-5 sectors likely to outperform today
2. Specific ETFs for each recommended sector
3. Brief rationale for each selection

## Analysis Framework

### Step 1: Market Context Assessment
- Review overnight global market performance (Asia, Europe)
- Check pre-market futures (S&P 500, Nasdaq, Dow)
- Identify major economic data releases scheduled for today
- Note significant corporate earnings reports
- Review recent Fed commentary or policy developments
- Assess geopolitical events impacting markets

### Step 2: Sector Analysis
Evaluate these factors for each sector:
- Recent price momentum (1-day, 5-day, 20-day trends)
- Sector rotation patterns
- Correlation with current macro themes
- Relative strength vs. S&P 500

### Step 3: ETF Selection
For each recommended sector, suggest liquid ETFs (volume > 1M shares)

## Output Format

**Market Overview:**
[2-3 sentences on overall market sentiment]

**Sector Recommendations:**

**1. [Sector Name] - [Bullish/Cautiously Bullish]**
- **Primary ETF:** [Ticker] - [ETF Name]
- **Alternative:** [Ticker] - [ETF Name]
- **Rationale:** [Why this sector should outperform today]
- **Key Catalysts:** [Specific events or trends]

[Repeat for 3-5 sectors]

**Sectors to Avoid:**
- [Sector]: [Brief reason]

**Risk Factors to Monitor:**
- [List 2-3 key risks]

**Trading Notes:**
- Position sizing approach
- Key levels to watch
- Timing considerations
"""

def get_market_analysis():
    """Call Claude API for market analysis"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    today = datetime.now().strftime('%A, %B %d, %Y')
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            temperature=1.0,
            messages=[{
                "role": "user",
                "content": MARKET_PROMPT.format(date=today)
            }]
        )
        
        return message.content[0].text
    
    except Exception as e:
        return f"Error getting analysis: {str(e)}"

def send_email(subject, body):
    """Send email via Gmail"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        
        # Add body
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"Email sent successfully at {datetime.now()}")
        return True
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def main():
    """Main execution function"""
    print(f"Starting market analysis at {datetime.now()}")
    
    # Get analysis from Claude
    analysis = get_market_analysis()
    
    # Create email subject
    today = datetime.now().strftime('%A, %B %d, %Y')
    subject = f"Daily Market Analysis - {today}"
    
    # Send email
    send_email(subject, analysis)
    
    # Optionally save to file as backup
    filename = f"market_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(filename, 'w') as f:
        f.write(analysis)
    print(f"Analysis saved to {filename}")

if __name__ == "__main__":
    main()

