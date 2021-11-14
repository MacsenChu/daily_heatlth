# Tensorflow+selenium实现识别验证码自动打卡

环境：

- Python 3.9

- Tesnorflow 1/2都行(如果是Tensorflow1.x 改成import tensorflow as tf即可)

目录结构：

- captcha ---- 截取的教务网登录验证码
- ckpt ---- 训练好的模型
- GenPics ---- 训练集
- GenPics_test ---- 测试集
- mail_html ---- 日报上传成功后会发送成功或失败的邮件html
- captcha_predict.py ---- 深度学习 卷积神经网络程序
- daily_health.py ---- 提交健康日报主程序

运行方式：
- 先修改daily_health.py里的学号、密码，接收邮件的邮箱，以及SMTP发送邮件邮箱的账户和授权码
- 再根据实际情况 修改好payload，即要提交的健康信息
- 然后执行指令 python daily_health.py

---

流程就是 在教务网登录后拿到登录状态的Cookie，然后带着Cookie直接POST表单地址。

![1581636855226_.pic](/Users/macsen/Desktop/1581636855226_.pic.jpg)

这个教务网登录的表单有极多的隐藏域，所以采用selenium模拟人工输入，验证码则采用卷积神经网络，训练好模型后去识别。



神经网络我用构建了三层：两层卷积层，一层全连接层。我看网上有好多大佬们能卷个五六层，这里要是卷那么多层 图就卷没了。



前面读取数据、解析csv文件不予列举了，(偷偷的说)训练集是我从教务网上爬了几千个验证码…

![WX20211114-135638@2x](/Users/macsen/Desktop/WX20211114-135638@2x.png)

### 构建卷积神经网络->y_predict

```python
def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))

```

### 构造损失函数、优化损失、计算准确率、开启会话、开启线程

原始验证码的大小是24 * 60，3通道，数量先为None

#### 第一个卷积大层

经过第一层卷积后 [None, 24, 60, 3] ---> [None,12, 30, 32]

```python
with tf.variable_scope("conv1"):
        # 卷积层
        # 定义filter和偏置
        conv1_weights = create_weights(shape=[5, 5, 3, 32])
        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
```



#### 第二个卷积大层

经过第二层卷积后 [None,12, 30, 3] ---> [None, 6, 15, 64]

```python
with tf.variable_scope("conv2"):
        # [None, 24, 60, 3] --> [None, 12, 30, 32]
        # 卷积层
        # 定义filter和偏置
        conv2_weights = create_weights(shape=[5, 5, 32, 64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
```

#### 全连接层

经过全连接层[None,  6 * 15 * 64] * [6 * 15 * 64,  4 * 10] = [None,  4 * 10]

```python
 with tf.variable_scope("full_connection"):
        # [None, 12, 30, 32] -> [None, 6, 15, 64]
        # [None, 6, 15, 64] -> [None, 6 * 15 * 64]
        # [None, 6 * 15 * 64] * [6 * 15 * 64, 4 * 10] = [None, 4 * 10]
        x_fc = tf.reshape(pool2_x, shape=[-1, 6 * 15 * 64])
        weights_fc = create_weights(shape=[6 * 15 * 64, 4 * 10])
        bias_fc = create_weights(shape=[4 * 10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc
```



#### 整个构建模型的函数

```python
def create_model(x):
    """
    构建卷积神经网络
    :param x:[None, 24, 60, 3]
    :return:
    """
    # 1）第一个卷积大层
    with tf.variable_scope("conv1"):
        # 卷积层
        # 定义filter和偏置
        conv1_weights = create_weights(shape=[5, 5, 3, 32])
        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2）第二个卷积大层
    with tf.variable_scope("conv2"):
        # [None, 24, 60, 3] --> [None, 12, 30, 32]
        # 卷积层
        # 定义filter和偏置
        conv2_weights = create_weights(shape=[5, 5, 32, 64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3）全连接层
    with tf.variable_scope("full_connection"):
        # [None, 12, 30, 32] -> [None, 6, 15, 64]
        # [None, 6, 15, 64] -> [None, 6 * 15 * 64]
        # [None, 6 * 15 * 64] * [6 * 15 * 64, 4 * 10] = [None, 4 * 10]
        x_fc = tf.reshape(pool2_x, shape=[-1, 6 * 15 * 64])
        weights_fc = create_weights(shape=[6 * 15 * 64, 4 * 10])
        bias_fc = create_weights(shape=[4 * 10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict
```

这样就构建好了卷积神经网络模型

### 开始训练模型

```python
def captcha_train():
    filename, image = read_picture()
    csv_data = parse_csv()

    # 1、准备数据
    x = tf.placeholder(tf.float32, shape=[None, 24, 60, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 4 * 10])

    # 2、构建模型
    y_predict = create_model(x)

    # 3、构造损失函数
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_list)

    # 4、优化损失
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 5、计算准确率
    equal_list = tf.reduce_all(
        tf.equal(tf.argmax(tf.reshape(y_predict, shape=[-1, 4, 10]), axis=2),
                 tf.argmax(tf.reshape(y_true, shape=[-1, 4, 10]), axis=2)), axis=1)
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 创建一个saver
    saver = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:

        # 初始化变量
        sess.run(init)

        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if os.path.exists(os.path.join(dir_path, "ckpt", "checkpoint")):
            saver.restore(sess, os.path.join(dir_path, "ckpt", "model"))

        for i in range(1000):
            filename_value, image_value = sess.run([filename, image])
            # print("filename_value:\n", filename_value)
            # print("image_value:\n", image_value)

            labels = filename2label(filename_value, csv_data)
            # 将标签值转换成one-hot
            labels_value = tf.reshape(tf.one_hot(labels, depth=10), [-1, 4 * 10]).eval()

            _, error, accuracy_value = sess.run([optimizer, loss, accuracy],
                                                feed_dict={x: image_value, y_true: labels_value})

            print("第%d次训练后损失为%f，准确率为%f" % (i + 1, error, accuracy_value))
            if i % 100 == 0:
                saver.save(sess, "./ckpt/model")

        # 回收线程
        coord.request_stop()
        coord.join(threads)
```

我优化了一些参数，第一百多批次训练后就已经出现100%的准确率了，测试集的表现也在99%以上。

![image-20211114125346935](/Users/macsen/Desktop/image-20211114125346935.png)



### 验证码识别

随便找几个验证码 也是手拿把掐

![WX20211114-134534@2x](/Users/macsen/Desktop/WX20211114-134534@2x.png)



### selenium获取教务网登录状态

此时就可以使用selenium对教务网下手。这里有个坑，selenium截的图是png格式的，png有4个通道，我训练的模型是jpg的，3个通道，需要进行转换。然后就是正常的预测，拿到验证码，登录。

登录失败返回的Cookie列表里只有一个字典，登录成功返回的列表里有两个字典，所以在这里捕获一下越界异常，如果列表越界了那么说明登录失败，发送失败的通知邮件。

```python
def get_jsession_id():
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
```

拿到Cookie之后， 关闭selenium 直接用requests模块POST健康日报表单地址

### requests提交日报信息

在这里改用requests模块有两点原因，一是登录时候 如果用selenium 还需要操作个几步才能提交，二是在提交日报的页面里需要获取当前位置，我是放阿里云上运行的，阿里云在北京 那肯定不行，而且服务器端的chrome也未必能支持获取位置(要是真获取位置成功提交上 那我估计就要被谈话了)。

```python
def post_message():
    jeesite_session_id, JSESSIONID = get_jsession_id()
    # 构造请求头
    headers = {}
    # 构造请求体
    payload = {}
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
```

### Headers和POST的表单数据

这里列举一下拿到登录状态的Cookies后，构造的Header和发送的数据

```python
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

# jeesite_session_id JSESSIONID 在登录拿到的Cookie里

payload = {
        'edate': time.strftime("%Y-%m-%d", time.localtime()),
        'province': '辽宁省:大连市',
        'location': '中国辽宁省大连市甘井子区柳树南街',
        'location_cor': '38.******, 121.******', 
        'location_code': '',
        'jkzt': '身体状况良好',
        'is_hot': '否',
        'sfjrgr': '是',
        'sfqghb': '未被列为中高风险地区',
        'sfgl': '未被隔离'
    }
# 'location_cor'是经纬度，自己根据实际情况填
```



### 发送邮件

```python
def send_mail(msg, header):
    mail_host = "smtp.qq.com"
    mail_user = "----发送邮件的邮箱----"
    mail_pass = "----发送邮件邮箱的授权码----"

    sender = "----发送者名字----"
    receivers = ['---接收邮箱----']

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
        # 连接到服务器 有些云服务器封掉了25端口，改587端口即可
        smtp.connect(mail_host, 25)
        # 登录到服务器
        smtp.login(mail_user, mail_pass)
        # 发送
        smtp.sendmail(sender, receivers, message.as_string())
        # 退出
        smtp.quit()
    except smtplib.SMTPException as e:
        print('error', e)  # 打印错误
```

### 测试

![1621636867304_.pic](/Users/macsen/Desktop/1621636867304_.pic.jpg)

![1631636867305_.pic](/Users/macsen/Desktop/1631636867305_.pic.jpg)

### CentOS定时运行

#### 安装 crontabs服务并设置开机自启：

```
yum install crontabs
systemctl enable crond
systemctl start crond
```

#### 配置定时规则

```
vim /etc/crontab
```

#### 在配置文件中配置你的定时执行规则

```
0 8 * * * python /root/daily_health/daily_health.py
```

这样每天八点就会自动进行打卡

请大家一定要遵守防疫管控的各项规定，配合并听从各项措施和要求，保证上报信息的及时和准确。

