"""
Notification Service
Sends alerts via email, Slack, and push notifications
"""
import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx
import os

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Notification Service",
    description="Alert notifications via multiple channels",
    version="1.0.0"
)


# ============================================
# MODELS
# ============================================

class NotificationType(str, Enum):
    SIGNAL = "SIGNAL"
    ORDER_EXECUTED = "ORDER_EXECUTED"
    TARGET_HIT = "TARGET_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    DAILY_SUMMARY = "DAILY_SUMMARY"
    RISK_ALERT = "RISK_ALERT"
    SYSTEM_ALERT = "SYSTEM_ALERT"


class Channel(str, Enum):
    EMAIL = "EMAIL"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"


class NotificationRequest(BaseModel):
    type: NotificationType
    title: str
    message: str
    data: Optional[Dict] = None
    channels: List[Channel] = [Channel.EMAIL]
    priority: str = "normal"  # low, normal, high, urgent


class Notification(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: NotificationType
    title: str
    message: str
    data: Optional[Dict] = None
    channels: List[Channel]
    priority: str
    status: str = "pending"  # pending, sent, failed
    sent_at: Optional[datetime] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================
# NOTIFICATION SENDER
# ============================================

class NotificationSender:
    """Sends notifications via multiple channels"""
    
    def __init__(self):
        self.notifications: Dict[str, Notification] = {}
        
        # Email config
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "alerts@indian-ai-trader.com")
        self.to_email = os.getenv("ALERT_EMAIL", "")
        
        # Slack config
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        
        # Custom webhook
        self.custom_webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL", "")
    
    async def send(self, request: NotificationRequest) -> Notification:
        """Send notification via specified channels"""
        notification = Notification(
            type=request.type,
            title=request.title,
            message=request.message,
            data=request.data,
            channels=request.channels,
            priority=request.priority
        )
        
        self.notifications[notification.id] = notification
        
        errors = []
        
        for channel in request.channels:
            try:
                if channel == Channel.EMAIL:
                    await self._send_email(notification)
                elif channel == Channel.SLACK:
                    await self._send_slack(notification)
                elif channel == Channel.WEBHOOK:
                    await self._send_webhook(notification)
            except Exception as e:
                errors.append(f"{channel}: {str(e)}")
                logger.error(f"Failed to send via {channel}: {e}")
        
        if errors:
            notification.status = "partial" if len(errors) < len(request.channels) else "failed"
            notification.error = "; ".join(errors)
        else:
            notification.status = "sent"
        
        notification.sent_at = datetime.utcnow()
        return notification
    
    async def _send_email(self, notification: Notification):
        """Send email notification"""
        if not self.smtp_user or not self.to_email:
            raise ValueError("Email not configured")
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{notification.type.value}] {notification.title}"
        msg["From"] = self.from_email
        msg["To"] = self.to_email
        
        # Plain text version
        text = f"{notification.title}\n\n{notification.message}"
        if notification.data:
            text += f"\n\nDetails:\n{notification.data}"
        
        # HTML version
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: #16213e; border-radius: 8px; padding: 20px;">
                <h2 style="color: #3b82f6; margin-top: 0;">{notification.title}</h2>
                <p style="font-size: 16px; line-height: 1.6;">{notification.message}</p>
                {self._format_data_html(notification.data) if notification.data else ''}
                <hr style="border: 1px solid #0f3460; margin: 20px 0;">
                <p style="color: #888; font-size: 12px;">Indian AI Trader - {notification.type.value}</p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))
        
        # Send email
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
        
        logger.info(f"Email sent: {notification.title}")
    
    def _format_data_html(self, data: Dict) -> str:
        """Format data as HTML table"""
        if not data:
            return ""
        
        rows = "".join(
            f'<tr><td style="padding: 8px; border-bottom: 1px solid #0f3460;">{k}</td>'
            f'<td style="padding: 8px; border-bottom: 1px solid #0f3460; font-family: monospace;">{v}</td></tr>'
            for k, v in data.items()
        )
        
        return f'''
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            {rows}
        </table>
        '''
    
    async def _send_slack(self, notification: Notification):
        """Send Slack notification"""
        if not self.slack_webhook_url:
            raise ValueError("Slack webhook not configured")
        
        # Build Slack message
        color = {
            NotificationType.SIGNAL: "#3b82f6",
            NotificationType.ORDER_EXECUTED: "#10b981",
            NotificationType.TARGET_HIT: "#22c55e",
            NotificationType.STOP_LOSS_HIT: "#ef4444",
            NotificationType.RISK_ALERT: "#f59e0b",
            NotificationType.SYSTEM_ALERT: "#6366f1",
        }.get(notification.type, "#888888")
        
        payload = {
            "attachments": [{
                "color": color,
                "title": notification.title,
                "text": notification.message,
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in (notification.data or {}).items()
                ],
                "footer": f"Indian AI Trader | {notification.type.value}",
                "ts": int(notification.created_at.timestamp())
            }]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
        
        logger.info(f"Slack notification sent: {notification.title}")
    
    async def _send_webhook(self, notification: Notification):
        """Send to custom webhook"""
        if not self.custom_webhook_url:
            raise ValueError("Custom webhook not configured")
        
        payload = {
            "id": notification.id,
            "type": notification.type.value,
            "title": notification.title,
            "message": notification.message,
            "data": notification.data,
            "priority": notification.priority,
            "timestamp": notification.created_at.isoformat()
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.custom_webhook_url, json=payload)
            response.raise_for_status()
        
        logger.info(f"Webhook notification sent: {notification.title}")
    
    def get_notifications(self, limit: int = 100) -> List[Notification]:
        """Get recent notifications"""
        notifs = list(self.notifications.values())
        notifs.sort(key=lambda x: x.created_at, reverse=True)
        return notifs[:limit]


# Global sender
sender = NotificationSender()


# ============================================
# HELPER FUNCTIONS FOR COMMON NOTIFICATIONS
# ============================================

async def notify_signal(
    symbol: str,
    action: str,
    confidence: float,
    entry: float,
    stop_loss: float,
    target: float
):
    """Send signal notification"""
    await sender.send(NotificationRequest(
        type=NotificationType.SIGNAL,
        title=f"New {action} Signal: {symbol}",
        message=f"AI generated a {action} signal for {symbol} with {confidence*100:.0f}% confidence",
        data={
            "Symbol": symbol,
            "Action": action,
            "Confidence": f"{confidence*100:.0f}%",
            "Entry": f"₹{entry:,.2f}",
            "Stop Loss": f"₹{stop_loss:,.2f}",
            "Target": f"₹{target:,.2f}"
        },
        channels=[Channel.EMAIL, Channel.SLACK],
        priority="high" if confidence > 0.8 else "normal"
    ))


async def notify_order_executed(
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    order_id: str
):
    """Send order execution notification"""
    await sender.send(NotificationRequest(
        type=NotificationType.ORDER_EXECUTED,
        title=f"Order Executed: {action} {symbol}",
        message=f"{action} {quantity} shares of {symbol} at ₹{price:,.2f}",
        data={
            "Symbol": symbol,
            "Action": action,
            "Quantity": quantity,
            "Price": f"₹{price:,.2f}",
            "Order ID": order_id
        },
        channels=[Channel.EMAIL],
        priority="normal"
    ))


# ============================================
# ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "email_configured": bool(sender.smtp_user),
        "slack_configured": bool(sender.slack_webhook_url),
        "total_notifications": len(sender.notifications)
    }


@app.post("/send")
async def send_notification(
    request: NotificationRequest,
    background_tasks: BackgroundTasks
):
    """Send a notification"""
    notification = await sender.send(request)
    return notification


@app.get("/notifications")
async def get_notifications(limit: int = 100):
    """Get recent notifications"""
    return {"notifications": sender.get_notifications(limit)}


@app.post("/test")
async def test_notification(channel: Channel = Channel.EMAIL):
    """Send a test notification"""
    notification = await sender.send(NotificationRequest(
        type=NotificationType.SYSTEM_ALERT,
        title="Test Notification",
        message="This is a test notification from Indian AI Trader",
        data={"Test": "Success", "Time": datetime.utcnow().isoformat()},
        channels=[channel],
        priority="low"
    ))
    return notification


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=True)
