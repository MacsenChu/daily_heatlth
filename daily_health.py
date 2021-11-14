import os
import requests
import captcha_predict
from selenium import webdriver
from PIL import Image
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from string import Template

url = "https://yiban.lnnu.edu.cn/form/index.php"
post_url = "https://yiban.lnnu.edu.cn/form/api/submitForm.php"
dir_path = os.path.dirname(os.path.abspath(__file__))


def get_jsession_id():
    """
    获取jession_id，即登录状态
    :return:
    """
    # 设置selenium一大堆参数
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36")
    browser = webdriver.Chrome(options=options)

    # 请求教务网的登录页面
    browser.get(url)
    browser.find_element_by_name("username").send_keys("----你的学号----")
    browser.find_element_by_name("password").send_keys("----你的密码----")
    browser.find_element_by_name("pimg").screenshot(os.path.join(dir_path, "captcha", "captcha.png"))

    # 这里截的图是png格式，png是四个通道，我的神经网络训练的三通道的，需要转成jpg
    captcha_image = Image.open(os.path.join(dir_path, "captcha", "captcha.png"))
    captcha_image = captcha_image.convert("RGB")
    captcha_image.save(os.path.join(dir_path, "captcha", "captcha.jpg"), quality=95)

    # 开始识别
    captcha_str = captcha_predict.captcha_predict(os.path.join(dir_path, "captcha", "captcha.jpg"))
    browser.find_element_by_name("j_digitPicture").send_keys(captcha_str)

    # 提交 获取Cookies
    browser.find_element_by_name("submit").click()
    cookies = browser.get_cookies()
    browser.quit()
    # 获取登录状态
    try:
        return cookies[1]["value"], cookies[0]["value"]
    except:
        with open(os.path.join(dir_path, "mail_html", "fail.html"), "r") as f:
            msg = f.read()
        msg = Template(msg)
        send_mail(msg.substitute(date=time.strftime("%Y-%m-%d", time.localtime())), "健康日报上传失败")


def send_mail(msg, header):
    """
    发送邮件
    :param msg:
    :param header:
    :return:
    """
    mail_host = "smtp.qq.com"
    mail_user = "smtp发送邮件的邮箱@qq.com"
    mail_pass = "授权码"

    sender = "发送者的邮箱@qq.com(同上面的mail_user)"
    receivers = ['接受者的邮箱@qq.com']

    # 邮件内容
    message = MIMEText(msg, 'html', 'utf-8')
    # 邮件主题
    message['Subject'] = Header(header, 'utf-8')
    # 发送方信息
    message['From'] = "Macsen Chu"
    # 接收方信息
    message['To'] = receivers[0]

    # 发送邮件
    try:
        smtp = smtplib.SMTP()
        # 连接到服务器
        smtp.connect(mail_host, 25)
        # 登录到服务器
        smtp.login(mail_user, mail_pass)
        # 发送
        smtp.sendmail(sender, receivers, message.as_string())
        # 退出
        smtp.quit()
    except smtplib.SMTPException as e:
        print('error', e)  # 打印错误


def post_message():
    """
    健康打卡
    :return:
    """
    jeesite_session_id, JSESSIONID = get_jsession_id()
    # 构造请求头
    headers = {
        "Accept": "text/html, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36",
        "Cookie": "jeesite.session.id=" + jeesite_session_id + "; JSESSIONID=" + JSESSIONID,
        "Host": "yiban.lnnu.edu.cn",
        "Origin": "https://yiban.lnnu.edu.cn",
        "Referer": "https://yiban.lnnu.edu.cn/form/content.php",
    }
    # 构造请求体
    payload = {
        'edate': time.strftime("%Y-%m-%d", time.localtime()),
        'province': '辽宁省:大连市',
        'location': '中国辽宁省大连市XXX区XXX街',
        'location_cor': 'xx.xxxxx, xxx.xxxxxx',  # 经纬度 根据实际情况填写
        'location_code': '',
        'jkzt': '身体状况良好',
        'is_hot': '否',
        'sfjrgr': '是',
        'sfqghb': '未被列为中高风险地区',
        'sfgl': '未被隔离'
    }
    # 直接POST那个健康日报的表单地址
    response = requests.request("POST", post_url, headers=headers, data=payload)
    # 将字符串转成字典
    res = eval(response.text)
    if res["status"] == "ok":
        with open(os.path.join(dir_path, "mail_html", "success.html"), "r") as f:
            msg = f.read()
        msg = Template(msg)
        send_mail(msg.substitute(date=time.strftime("%Y-%m-%d", time.localtime())), "健康日报上传成功")
    else:
        with open(os.path.join(dir_path, "mail_html", "fail.html"), "r") as f:
            msg = f.read()
        msg = Template(msg)
        send_mail(msg.substitute(date=time.strftime("%Y-%m-%d", time.localtime())), "健康日报上传失败")


if __name__ == "__main__":
    post_message()
