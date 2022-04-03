import sys
import main
from PyQt5.Qt import *
from main import Ui_widget
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from dataload import get_data


# 贪心算法
def tanXin(m, h, v):

    data = open("data.txt", "w+")
    start = time.time()
    arr = [(i, v[i] / h[i], h[i], v[i]) for i in range(len(h))]   # 计算权重, 整合得到一个数组
    arr.sort(key = lambda x: x[1], reverse = True)   # 按照list中的权重，从大到小排序,list.sort() list排序函数
    bagVal = 0
    bagList = [0] * len(h)

    for i, w, h, v in arr:
        if w <= m:   # 1 如果能放的下宝物，那就把宝物全放进去
            m -= h
            bagVal += v
            bagList[i] = 1
        else:   # 2 如果宝物不能完全放下，考虑放入部分宝物
            bagVal += m * w
            bagList[i] = 1
            break

    print('最大价值：', bagVal, file=data)
    print('解向量：', bagList, file=data)
    return bagVal, bagList


# 动态规划算法
def bagg(n, m, w, v):

    val = [[0 for j in range(m + 1)] for i in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if j < w[i - 1]:
                val[i][j] = val[i - 1][j]
            else:
                val[i][j] = max(val[i - 1][j], val[i - 1][j - w[i - 1]] + v[i - 1])   # 背包总容量够放当前物体，取最大价值

    return val

def dongTai(n, m, w, val):

    data = open("data.txt", "w+")
    bagList = [0] * len(w)
    x = [0 for i in range(n)]
    j = m

    # 求出解向量
    for i in range(n, 0, -1):
        if val[i][j] > val[i - 1][j]:
            x[i - 1] = 1
            j -= w[i - 1]
    for i in range(n):
        if x[i]:
            bagList[i] = 1

    print('最大价值为:', val[n][m], file=data)
    print('解向量：', bagList, file=data)
    return val[n][m], bagList


# 回溯算法
bestV = 0
curW = 0
curV = 0
bestx = None

def huiS(i):

    global bestV, curW, curV, bestx

    if i >= n:
        if bestV < curV:
            bestV = curV
            bestx = x[:]
    else:
        if curW + w[i] <= m:
            x[i] = 1
            curW += w[i]
            curV += v[i]
            huiS(i + 1)
            curW -= w[i]
            curV -= v[i]
        x[i] = 0
        huiS(i + 1)


# 遗传算法
def init(N,n):
    C = []
    for i in range(N):
        c = []
        for j in range(n):
            a = np.random.randint(0,2)
            c.append(a)
        C.append(c)
    return C


##评估函数
# x(i)取值为1表示被选中，取值为0表示未被选中
# w(i)表示各个分量的重量，v（i）表示各个分量的价值，w表示最大承受重量
def fitness(C,N,n,W,V,w):
    S = []##用于存储被选中的下标
    F = []## 用于存放当前该个体的最大价值
    for i in range(N):
        s = []
        h = 0  # 重量
        f = 0  # 价值
        for j in range(n):
            if C[i][j]==1:
                if h+W[j]<=w:
                    h=h+W[j]
                    f = f+V[j]
                    s.append(j)
        S.append(s)
        F.append(f)
    return S,F

##适应值函数,B位返回的种族的基因下标，y为返回的最大值
def best_x(F,S,N):
    y = 0
    x = 0
    B = [0]*N
    for i in range(N):
        if y<F[i]:
            x = i
        y = F[x]
        B = S[x]
    return B,y

## 计算比率
def rate(x):
    p = [0] * len(x)
    s = 0
    for i in x:
        s += i
    for i in range(len(x)):
        p[i] = x[i] / s
    return p

## 选择
def chose(p, X, m, n):
    X1 = X
    r = np.random.rand(m)
    for i in range(m):
        k = 0
        for j in range(n):
            k = k + p[j]
            if r[i] <= k:
                X1[i] = X[j]
                break
    return X1

##交配
def match(X, m, n, p):
    r = np.random.rand(m)
    k = [0] * m
    for i in range(m):
        if r[i] < p:
            k[i] = 1
    u = v = 0
    k[0] = k[0] = 0
    for i in range(m):
        if k[i]:
            if k[u] == 0:
                u = i
            elif k[v] == 0:
                v = i
        if k[u] and k[v]:
            # print(u,v)
            q = np.random.randint(n - 1)
            # print(q)
            for i in range(q + 1, n):
                X[u][i], X[v][i] = X[v][i], X[u][i]
            k[u] = 0
            k[v] = 0
    return X

##变异
def vari(X, m, n, p):
    for i in range(m):
        for j in range(n):
            q = np.random.rand()
            if q < p:
                X[i][j] = np.random.randint(0,2)

    return X


# 散点图
def sanDt(w, v):

    plt.title("value-weight", fontsize = 15)   # 图名称
    plt.xlabel("weight", fontsize = 8)
    plt.ylabel("value", fontsize = 8)
    plt.axis([0, 100, 0, 100])   # 设置x,y轴长度
    plt.scatter(w, v, s = 20)   # 将数据传入x,y轴
    plt.savefig('sanD.jpg')   # 保存图片
    plt.close()


# 打开选择的测试数据，调用选择的算法，并从图形界面输出结果
def data_select():

    global ui, m, n, w, v
    (t, t1) = display()   # 获取文本框输入值
    t = str(t)
    sets = get_data(t)

    w_in = []  # 物品重量
    v_in = []  # 物品价值
    for i in sets:
        n = i['size'] # 物品个数
        m = i['capacity'] # 背包容量
        w_in =i['weight']
        v_in =i['profit']
    str_w = [i for i in w_in.split(',')]
    str_v = [i for i in v_in.split(',')]
    w = list(map(int, str_w))
    v = list(map(int, str_v))

    s = int(t1)
    if s == 1:   # s == 1，贪心算法
        start = time.time()
        (bagVal, bagList) = tanXin(m, w, v)
        end = time.time()
        data = open("data.txt", "a+")
        print("运行时间", end - start, "s", file=data)
        data.close()
        value = '最大价值：' + str(bagVal) + '\n' + '解向量：' + str(bagList) + \
                '\n' + '运行时间：' + str(end - start)
    elif s == 2:   # s == 2，动态规划算法
        start = time.time()
        val = bagg(n, m, w, v)
        (bagVal, bagList) = dongTai(n, m, w, val)
        end = time.time()
        data = open("data.txt", "a+")
        print("运行时间", end - start, "s", file = data)
        data.close()
        value = '最大价值：' + str(bagVal) + '\n' + '解向量：' + str(bagList) +\
                '\n' + '运行时间：'  + str(end - start)
    elif s == 3:
        global x
        data = open("data.txt", "w+")
        start = time.time()
        x = [False for i in range(n)]
        huiS(0)
        bagVal = bestV
        print("最大价值：", bagVal, file=data)
        print("解向量：", bestx, file=data)
        end = time.time()
        print("运行时间", end - start, "s", file=data)
        data.close()
        value = '最大价值：' + str(bagVal) + '\n' + '解向量：' + str(bestx) + \
                '\n' + '运行时间：' + str(end - start)
    elif s == 4:
        start = time.time()
        size = 8  # 规模
        N = 2 * n  # 迭代次数
        Pc = 0.8  # 交配概率
        Pm = 0.05  # 变异概率
        C = init(size, n)
        S, F = fitness(C, size, n, w, v, m)
        B, bagVal = best_x(F, S, size)
        Y = [bagVal]
        for i in range(N):
            p = rate(F)
            C = chose(p, C, size, n)
            C = match(C, size, n, Pc)
            C = vari(C, size, n, Pm)
            S, F = fitness(C, size, n, w, v, m)
            B1, y1 = best_x(F, S, size)
            if y1 > bagVal:
                bagVal = y1
            Y.append(bagVal)
        end = time.time()
        value = '最大价值：' + str(bagVal) + '\n' + '解向量：' + str(C[0]) + \
                '\n' + '运行时间：'  +str(end - start)

    sanDt(w, v)
    pix = QPixmap('sanD.jpg')
    ui.textEdit.setPlainText('物品重量'+str(w_in)+'\n'+'物品价值'+str(v_in))
    ui.textEdit_2.setPlainText(value)
    ui.label_3.setScaledContents(True)
    ui.label_3.setPixmap(pix)
    table_text(bagVal, end - start)


# 生成运行记录
def table_text(bagVal, runTime):
    global ui
    nowTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ui.tableWidget.setItem(ui.i, 0, QTableWidgetItem(ui.comboBox.currentText()))
    ui.tableWidget.setItem(ui.i, 1, QTableWidgetItem(str_x))
    ui.tableWidget.setItem(ui.i, 2, QTableWidgetItem(str(bagVal)))
    ui.tableWidget.setItem(ui.i, 3, QTableWidgetItem(str(runTime)))
    ui.tableWidget.setItem(ui.i, 4, QTableWidgetItem(str(nowTime)))
    ui.i += 1


# 获取选择的算法，生成相应的值
def radio_toggled(btn):
    global radiox, str_x
    if btn.text() == '贪心算法':
        radiox = 1
        str_x = '贪心算法'
    elif btn.text() == '动态规划算法':
        radiox = 2
        str_x = '动态规划算法'
    elif btn.text() == '回溯算法':
        radiox = 3
        str_x = '回溯算法'
    elif btn.text() == '遗传算法':
        radiox = 4
        str_x = '遗传算法'


# 获取数据和算法的值
def display():
    global ui
    data = ui.comboBox.currentIndex()
    data2 = radiox
    return data, data2


def com_clicked():
    global ui
    ui.tabWidget.setCurrentIndex(ui.tabWidget.currentIndex() + 1)

# 主函数
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = main.Ui_widget()
    MainWindow = QMainWindow()
    ui.setupUi(MainWindow)
    ui.radioButton.toggled.connect(lambda: radio_toggled(ui.radioButton))
    ui.radioButton_2.toggled.connect(lambda: radio_toggled(ui.radioButton_2))
    ui.radioButton_3.toggled.connect(lambda: radio_toggled(ui.radioButton_3))
    ui.radioButton_4.toggled.connect(lambda: radio_toggled(ui.radioButton_4))
    ui.pushButton.clicked.connect(data_select)
    ui.i = 0
    ui.commandLinkButton.clicked.connect(com_clicked)
    ui.commandLinkButton_2.clicked.connect(com_clicked)
    ui.commandLinkButton_3.clicked.connect(com_clicked)
    ui.commandLinkButton_4.clicked.connect(com_clicked)
    ui.commandLinkButton_5.clicked.connect(com_clicked)
    MainWindow.show()
    sys.exit(app.exec_())