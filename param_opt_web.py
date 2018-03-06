import sys
sys.path.append(".")
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import json
from opts.optexecutor import OptExecutorFactory
from utils.checker_jsonstructure import WrongStructureException

# instance name:param_opt_web
param_opt_web = Flask(__name__)


def picked_up():
    messages = [
        "will prepare json message examples..."
    ]
    return np.random.choice(messages)


# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理
@param_opt_web.route('/')
def index():
    title = "Hyper Parameter Search by Tree-parzen Estimator"
    message = picked_up()
    # index.html をレンダリングする
    return render_template('index.html',
                           message=message, title=title)

# http://127.0.0.1:5000/suggest?condition=hoge
@param_opt_web.route('/suggest', methods=['GET', 'POST'])
def suggest():
    if request.method == 'GET':
        condition = request.args.get('condition')
        # print(condition)
        # condition = '{"seed":0,"lib":"hyperopt","algo":"tpe","scope":{"x":["uniform",-10,10],"y":["uniform",-10,10]},"max_evals":1,"results":{"losses":[3.4777,3.294,28.729,19.62,20.458],"statuses":["ok","ok","ok","ok","ok"],"vals":{"y":[-0.13974,0.3722,-2.419,0.28755,-3.2827],"x":[1.8571,1.765,4.615,-4.068,2.832]}}}'
        # print(condition)
    elif request.method == 'POST':
        condition = request.json
    else:
        return "Unexpected request method {}.".format(request.method)

    try:
        executor = OptExecutorFactory.get_executor(condition)
        rval = executor.suggest()
    except WrongStructureException as err:
        print("Exception {}".format(err))
        # rval = dict(statuses=err.args)
        import traceback

        rval = dict(statuses=[err.args])


    render_template('index.html', title='Parameter Optimizer Condition', message=rval)
    print(type(rval))
    return json.dumps(rval)


if __name__ == '__main__':
    param_opt_web.debug = True # デバッグモード有効化
    # param_opt_web.debug = False
    # param_opt_web.run(host='127.0.0.1')
    param_opt_web.run(host='JP04990730')