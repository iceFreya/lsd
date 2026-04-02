import yfinance as yf
import yagmail

def get_stock_price(symbol):
    try:
        stock_price = yf.Ticker(symbol)
        last_price = stock_price.history(period='1d')['Close'].iloc[-1]
        return last_price
    except Exception as e:
        print(f"wrong:{e}")

def send_email(to, content):
    sender_email = "发送对象的邮箱"
    sender_auth_code = "发送对象的邮箱授权码"
    server_smtp = "smtp.qq.com"

    try:
        yag = yagmail.SMTP(sender_email, sender_auth_code, server_smtp)
        yag.send(to=to, content=content)
        return True
    except Exception as e:
        print(f"wrong:{e}")


