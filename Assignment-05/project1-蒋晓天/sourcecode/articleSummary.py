from summaryModel import SummaryModel
from flask import Flask, request, redirect, url_for, render_template

# 模型初始化
m = SummaryModel()
# 用来记录输入和输出信息
d = {}

app = Flask(__name__)

# 接受输入，并重定向给输出页面
@app.route('/alex/input/', methods=['POST', 'GET'])
def getdata():
    if request.method == 'POST':
        d['title'] = request.form.get('title')
        d['content'] = request.form['content']
        return redirect(url_for('result'))
    return render_template('input.html')

# 对输入进行计算，得到摘要后显示相应的输出页面
@app.route('/alex/summary')
def result():
    summary = m.get_summary(d['title'], d['content'])
    d['summary'] = summary
    return render_template('result.html', summary=d['summary'], title=d['title'], content=d['content'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5080)