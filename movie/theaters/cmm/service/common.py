class Common(object):
    def __init__(self):
        pass

    @staticmethod
    def menu(ls):
        for i, j in enumerate(ls):  # ["종료", "입력", "출력", "삭제"]
            print(f'{i}. {j}')
        return input("메뉴 선택 : ")
