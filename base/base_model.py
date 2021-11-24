class BaseModel(Object):
    def __init__(self, initial_input=False, *args):
        """
        在这里指定模型参数如层数、size等
        如果需要初始化输入，在这里初始化输入
        if initial_input:
            self.input_x = tf.placeholder(...)
        """
        pass
    
    def forward(self, *args):
        """
        在这里定义计算图
        self.output = xxx
        """
        pass

    def cal_metrics(self, *args):
        """在这里定义指标
        self.loss = xxx
        self.accuracy = xxx
        """
        pass

    def build(self, **args):
        self.forward()
        self.cal_metrics()

    def show(self):
        """
        输出输入输出维度
        """
        pass