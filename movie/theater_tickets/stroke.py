import copy
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from matplotlib import font_manager, rc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

STROKE_MENUS = ["종료",  # 0
                "데이터 구조 파악(Spec)",  # 1
                "변수 한글화(Rename)",  # 2
                "연속형(Interval) 변수 편집",  # 3 18세 이상만 사용함
                "범주형(Norminal) 변수 편집",  # 4
                "타깃(Target)",  # 5
                "파티션(Partition)",  # 6
                "학습",  # 7
                "예측"]  # 8

stroke_meta = {
    'id': '아이디', 'gender': '성별', 'age': '나이',
    'hypertension': '고혈압',
    'heart_disease': '심장병',
    'ever_married': '기혼 여부',
    'work_type': '직종',
    'Residence_type': '거주 형태',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량 지수',
    'smoking_status': '흡연 여부',
    'stroke': '뇌졸중'
}
stroke_menu = {
    "1": lambda x: x.spec(),
    "2": lambda x: x.rename_meta(),
    "3": lambda x: x.interval_variables(),
    "4": lambda x: x.norminal_variables(),
    "5": lambda x: x.targetting(),
    "6": lambda x: x.partition(),
    "7": lambda x: x.hook()
}

'''
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 5110 non-null   int64  
 1   gender             5110 non-null   object 
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64  
 4   heart_disease      5110 non-null   int64  
 5   ever_married       5110 non-null   object 
 6   work_type          5110 non-null   object 
 7   Residence_type     5110 non-null   object 
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object 
 11  stroke             5110 non-null   int64  
dtypes: float64(3), int64(4), object(5)
'''


class StrokeService:
    def __init__(self):
        self.stroke = pd.read_csv('data/stroke/healthcare-dataset-stroke-data.csv')
        self.my_stroke = self.stroke.rename(columns=stroke_meta)
        self.adult_stroke = copy.deepcopy(self.my_stroke)
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def hook(self):
        self.rename_meta()
        self.interval_variables()
        self.norminal_variables()
        self.targetting()
        self.partition()
        self.learning(flag='gini')

    '''
   1.스펙 보기
   '''

    def spec(self):
        print(" --- 1.Shape ---")
        print(self.stroke.shape)
        print(" --- 2.Features ---")
        print(self.stroke.columns)
        print(" --- 3.Info ---")
        print(self.stroke.info())
        print(" --- 4.Case Top1 ---")
        print(self.stroke.head(1))
        print(" --- 5.Case Bottom1 ---")
        print(self.stroke.tail(3))
        print(" --- 6.Describe ---")
        print(self.stroke.describe())
        print(" --- 7.Describe All ---")
        print(self.stroke.describe(include='all'))

    '''
   2.한글 메타 데이터
   '''

    def rename_meta(self):
        print(" --- 2.Features ---")
        print(self.my_stroke.columns)

    '''
    3.타깃 변수(= 종속 변수 dependent, Y값) 설정
    입력 변수(= 설명 변수, 확률 변수, X값)
    타깃 변수명: stroke (=뇌졸중)
    타깃 변수값: 과거에 한 번이라도 뇌졸중이 발병했으면 1, 아니면 0
    인터벌 = ['나이','평균 혈당','체질량 지수']
    '''

    def interval_variables(self):
        interval = ['나이', '평균 혈당', '체질량 지수']
        print(f'--- 구간 변수 타입 ---\n{self.my_stroke[interval].dtypes}')
        stroke1 = self.my_stroke[self.my_stroke['나이'] > 18]
        print(f'--- 결측 값을 가진 column ---\n{stroke1.isna().any()[lambda x: x]}')
        print(f"--- 체질량 결측값 비율 확인 ---\n{self.my_stroke['체질량 지수'].isnull().mean()}")
        print(f'고유 값 개수 : {len(pd.unique(self.my_stroke["아이디"]))}')
        pd.options.display.float_format = '{:.2f}'.format
        print(f'--- 구간 변수 기초 통계량 --- \n{self.my_stroke[interval].describe()}')
        criterion = self.my_stroke['나이'] > 18
        self.adult_stoke = self.my_stroke[criterion]
        print(f'--- 성인 객체 스펙 --- \n{self.adult_stoke.shape}')
        # 평균 혈당 232.64이하와 체질량 지수 60.3이하를 이상치로 규정 하고 제거함
        self.adult_stoke
        c1 = self.adult_stoke['평균 혈당'] <= 232.64
        c2 = self.adult_stoke['체질량 지수'] <= 60.3
        self.adult_stoke = self.adult_stoke[c1 & c2]
        print(f'--- 이상치 제거한 성인 객체 스펙 ---\n{self.adult_stoke.shape}')

    '''
    4.범주형 = ['성별', '심장병', '기혼 여부', '직종', '거주 형태', '흡연 여부', '뇌졸중']
    '''

    def ratio_variables(self):  # 해당 칼럼이 없음
        pass

    def norminal_variables(self):
        category = ['성별', '심장병', '기혼 여부', '직종', '거주 형태', '흡연 여부', '고혈압']
        print(f'--- 범주형 변수 데이터 타입 ---\n{self.adult_stroke[category].dtypes}')
        print(f'--- 범주형 변수 결측값 ---\n{self.adult_stroke[category].isnull().sum()}')
        print(f'--- 결측 값 있는 변수 ---\n{self.adult_stroke[category].isna().any()[lambda x: x]}')
        self.adult_stroke['성별'] = OrdinalEncoder().fit_transform(self.adult_stroke['성별'].values.reshape(-1, 1))
        self.adult_stroke['기혼 여부'] = OrdinalEncoder().fit_transform(self.adult_stroke['기혼 여부'].values.reshape(-1, 1))
        self.adult_stroke['직종'] = OrdinalEncoder().fit_transform(self.adult_stroke['직종'].values.reshape(-1, 1))
        self.adult_stroke['거주 형태'] = OrdinalEncoder().fit_transform(self.adult_stroke['거주 형태'].values.reshape(-1, 1))
        self.adult_stroke['흡연 여부'] = OrdinalEncoder().fit_transform(self.adult_stroke['흡연 여부'].values.reshape(-1, 1))

        self.stroke = self.adult_stroke
        self.spec()
        self.stroke.to_csv('save/stroke/stroke.csv', index=False)
        print(" ### 프리 프로세스 종료 ### ")

    def ordinal_variables(self):  # 해당 칼럼이 없음
        pass

    def targetting(self):
        df = pd.read_csv('save/stroke/stroke.csv')
        self.data = df.drop(['뇌졸중'], axis=1)
        self.data = self.data.drop(['아이디'], axis=1) # 오버피팅 방지를 위해 아이디 삭제
        self.target = df['뇌졸중']
        print(f'--- data shape --- \n {self.data}')
        print(f'--- target shape --- \n {self.target}')

    def partition(self):
        data = self.data
        target = self.target
        undersample = RandomUnderSampler(sampling_strategy=0.333, random_state=2)
        data_under, target_under = undersample.fit_resample(self.data, self.target)
        print(target_under.value_counts(dropna=True))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_under, target_under, test_size=0.5, random_state=42,
                                                            stratify=target_under)

        print("X_train shape : ", self.X_train.shape)
        print("X_test shape : ", self.X_test.shape)
        print("y_train shape : ", self.y_train.shape)
        print("y_test shape : ", self.y_test.shape)

    def learning(self, flag):
        if flag == 'ccee':
            tree = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=5)
            tree.fit(self.X_train, self.y_train)
        else:
            tree = DecisionTreeClassifier(criterion="gini", random_state=0, max_depth=5)
            params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 21)}
            # 5회의 교차 검증을 2개의 기준마다 20개의 max_depth 값으로 대입 (5 * 2 * 20 = 총 500 회) 학습(fit) 됨
            grid_tree = GridSearchCV(
                tree,
                param_grid=params,
                scoring='accuracy',
                cv=5, # 교차 검증 파라미터 5회 실시
                n_jobs=-1, # 멀티코어 CPU 모두 사용
                verbose=1) # 연산중간 메시지 출력
            grid_tree.fit(self.X_train, self.y_train)
            tree.fit(self.X_train, self.y_train)

        print("GridSearchCV max accuracy:{:.5f}".format(grid_tree.best_score_))
        print("GridSearchCV best parameter: ", grid_tree.best_params_)

        best_clf = grid_tree.best_estimator_
        pred = best_clf.predict(self.X_test)
        print("Accuracy on test set:{:.5f}".format(accuracy_score(self.y_test, pred)))

        print("Feature importances:")
        print(best_clf.feature_importances_)

        feature_name = list(self.data.columns)
        dft = pd.DataFrame(np.round(best_clf.feature_importances_, 4),
                           index=feature_name, columns=['Feature_importances'])
        dft1 = dft.sort_values(by='Feature_importances', ascending=False)
        print(dft1)

        ax = sns.barplot(y=dft1.index, x="Feature_importances", data=dft1)

        for p in ax.patches:
            ax.annotate("%.3f" % p.get_width(), (p.get_x() + p.get_width(),
                                                 p.get_y()+1), xytext=(5, 10), textcoords='offset points')

        plt.show()

if __name__ == '__main__':
    def my_menu(ls):
        for i, j in enumerate(ls):
            print(f"{i}. {j}")
        return input('메뉴선택: ')


    t = StrokeService()
    while True:
        menu = my_menu(STROKE_MENUS)
        if menu == '0':
            print("종료")
            break
        else:
            stroke_menu[menu](t)
