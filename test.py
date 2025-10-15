from evaluator import Evaluator


def test():
    predict = "先上升后持平,涨幅约0.5%-2.5%"
    actual = "基本持平,0.5%以内"
    score = Evaluator.score(predict, actual)
    print(f"预测: {predict}\n实际: {actual}\n评分: {score}")

if __name__ == "__main__":
    test()