'''
데이터를 활용해서 서울시내 경찰서 범죄발생과
검거율 현황지도(폴리움)를 작성하시오.
'''
from dataclasses import dataclass
import folium
import googlemaps
import numpy as np
import pandas as pd
from sklearn import preprocessing

CRIME_MENUS = ['종료',
               'Show Spce',
               'Save police_pos',
               'Save cctv_pop',
               'Save police_norm',
               'Folium Example',
               'Folium police']

crime_menu = {
    "1": lambda x: x.crime_spec(),
    "2": lambda x: x.save_police_pos(),
    "3": lambda x: x.save_cctv_pop(),
    "4": lambda x: x.save_police_norm(),
    "5": lambda x: x.get_json_from_df(),
    "6": lambda x: x.save_seoul_folium()

}


@dataclass
class MyChoroplethVO:
    geo_data = "",
    data = object,
    name = "",
    columns = [],
    key_on = "",
    fill_color = "",
    fill_opacity = 0.0,
    line_opacity = 0.0,
    legend_name = "",
    bins = None,
    location = [],
    zoom_start = 0,
    save_path = ''


def MyChoroplethService(vo):
    map = folium.Map(location=vo.location, zoom_start=vo.zoom_start)
    folium.Choropleth(
        geo_data=vo.geo_data,
        data=vo.data,
        name=vo.name,
        columns=vo.columns,
        key_on=vo.key_on,
        fill_color=vo.fill_color,
        fill_opacity=vo.fill_opacity,
        line_opacity=vo.line_opacity,
        legend_name=vo.legend_name
    ).add_to(map)
    map.save(vo.save_path)


class CrimeSevrvice:
    def __init__(self):
        self.crime = pd.read_csv('data/crime/crime_in_seoul.csv')
        cols = ['절도 발생', '절도 검거', '폭력 발생', '폭력 검거']
        self.crime[cols] = self.crime[cols].replace(',', '', regex=True).astype(int)
        self.cctv = pd.read_csv('data/crime/cctv_in_seoul.csv')
        self.pop = pd.read_excel('data/crime/pop_in_seoul.xls', skiprows=[0, 2],
                                 usecols=['자치구', '합계', '한국인', '등록외국인', '65세이상고령자'])
        self.ls = [self.crime, self.cctv, self.pop]
        self.crime_rate_columns = ['살인 검거율', '강도 검거율', '강간 검거율', '절도 검거율', '폭력 검거율']
        self.crime_columns = ['살인', '강도', '강간', '절도', '폭력']
        self.arrest_columns = ['살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거']
        self.polic_pos = pd.read_pickle('save/crime/police-pos.pkl')
        self.us_states = 'data/crime/us-states.json'
        self.us_unemployment = pd.read_csv('data/crime/us_unemployment.csv')
        self.kr_states = 'data/crime/kr-state.json'

    def crime_spec(self):
        [(lambda x: print(f"--- 1.Shape ---\n{x.shape}\n"
                          f"--- 2.Features ---\n{x.columns}\n"
                          f"--- 3.Info ---\n{x.info}\n"
                          f"--- 4.Case Top1 ---\n{x.head(1)}\n"
                          f"--- 5.Case Bottom1 ---\n{x.tail(3)}\n"
                          f"--- 6.Describe ---\n{x.describe()}\n"
                          f"--- 7.Describe All ---\n{x.describe(include='all')}"))(i) for i in self.ls]

    '''
    2. 범죄율 합산 및 비율 칼럼 생성
    일부 데이터 정수형 변경 필요
    '''

    def save_police_pos(self):  # nominal
        crime = self.crime
        station_names = []
        for name in crime['관서명']:
            print(f"지역 이름 : {name[:-1]}")
            station_names.append(f'서울{str(name[:-1])}경찰서')
        print(f'--- 서울시 내 경찰서는 총 {len(station_names)}개 이다. ---')
        [print(f'경찰서 이름 : {str(i)}') for i in station_names]
        gmaps = (lambda x: googlemaps.Client(key=x))('')
        print(gmaps.geocode('서울중부경찰서', language="ko"))
        print('--- API에서 주소 추출 시작 ---')
        station_addrs = []
        station_lats = []
        station_lngs = []
        for i, name in enumerate(station_names):
            _ = gmaps.geocode(name, language="ko")
            print(f'name {i} = {_[0].get("formatted_address")}')
            station_addrs.append(_[0].get('formatted_address'))
            _loc = _[0].get('geometry')
            station_lats.append(_loc['location']['lat'])
            station_lngs.append(_loc['location']['lng'])
        gu_names = []
        for name in station_addrs:
            _ = name.split()
            gu_name = [gu for gu in _ if gu[-1] == '구'][0]
            gu_names.append(gu_name)
        crime['구별'] = gu_names
        # 구와 경찰서 위치가 다른 경우 수작업
        crime.loc[crime['관서명'] == '혜화서', ['구별']] == '종로구'
        crime.loc[crime['관서명'] == '서부서', ['구별']] == '은평구'
        crime.loc[crime['관서명'] == '강서서', ['구별']] == '양천구'
        crime.loc[crime['관서명'] == '종암서', ['구별']] == '성북구'
        crime.loc[crime['관서명'] == '방배서', ['구별']] == '서초구'
        crime.loc[crime['관서명'] == '수서서', ['구별']] == '강남구'
        crime.to_csv('./save/crime/crime_gu.csv', index=False)
        crime.to_pickle('./save/crime/police-pos.pkl')
        print('--- 저장 완료 ---')
        print('--- 데이터 로드 ---')
        print(pd.read_pickle('save/police-pos.pkl'))

    def crime_mean(self):
        category = ['살인 발생', '강도 발생', '강간 발생',
                    '절도 발생', '폭력 발생']
        self.my_crime[category] = pd.to_numeric(self.my_crime[category])
        self.my_crime['범죄 발생 수'] = self.my_crime[category].sum()
        print(self.my_crime.head(3))

    '''
    3. =
    '''

    def save_cctv_pop(self):  # ratio
        self.cctv.rename(columns={self.cctv.columns[0]: '구별'}, inplace=True)
        self.pop.rename(columns={
            self.pop.columns[0]: '구별',
            self.pop.columns[1]: '인구수',
            self.pop.columns[2]: '한국인',
            self.pop.columns[3]: '외국인',
            self.pop.columns[4]: '고령자',
        }, inplace=True)
        self.pop.drop(index=26, inplace=True)
        self.pop['외국인 비율'] = self.pop['외국인'].astype(int) / self.pop['인구수'].astype(int) * 100
        self.pop['고령자 비율'] = self.pop['고령자'].astype(int) / self.pop['인구수'].astype(int) * 100
        self.cctv.drop(['2013년도 이전', '2014년', '2015년', '2016년'], axis=1, inplace=True)
        cctv_pop = pd.merge(self.cctv, self.pop, on='구별')
        cor1 = np.corrcoef(cctv_pop['고령자 비율'], cctv_pop['소계'])
        cor2 = np.corrcoef(cctv_pop['외국인 비율'], cctv_pop['소계'])
        print(f'고령자 비율과 CCTV의 상관계수 {str(cor1)} \n'
              f'외국인 비율과 CCTV의 상관계수 {str(cor2)} ')
        """
         고령자비율과 CCTV 의 상관계수 [[ 1.         -0.28078554]
                                     [-0.28078554  1.        ]] 
         외국인비율과 CCTV 의 상관계수 [[ 1.         -0.13607433]
                                     [-0.13607433  1.        ]]
        r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
        r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
        r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
        r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
        r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
        r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
        r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계
        고령자비율 과 CCTV 상관계수 [[ 1.         -0.28078554] 약한 음적 선형관계
                                    [-0.28078554  1.        ]]
        외국인비율 과 CCTV 상관계수 [[ 1.         -0.13607433] 거의 무시될 수 있는
                                    [-0.13607433  1.        ]]                        
        """
        cctv_pop.to_pickle('save/crime/cctv_pop.pkl')
        print('--- 저장 완료 ---')
        print(pd.read_pickle('save/crime/cctv_pop.pkl'))

    def save_police_norm(self):
        polic_pos = self.polic_pos
        police = pd.pivot_table(polic_pos, index='구별', aggfunc=np.sum)
        police['살인 검거율'] = police['살인 검거'].astype(int) / police['살인 발생'].astype(int) * 100
        police['강도 검거율'] = police['강도 검거'].astype(int) / police['강도 발생'].astype(int) * 100
        police['강간 검거율'] = police['강간 검거'].astype(int) / police['강간 발생'].astype(int) * 100
        police['절도 검거율'] = police['절도 검거'].astype(int) / police['절도 발생'].astype(int) * 100
        police['폭력 검거율'] = police['폭력 검거'].astype(int) / police['폭력 발생'].astype(int) * 100
        police.drop(columns={'살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거'}, axis=1)

        police.rename(columns={
            '살인 발생': '살인',
            '강도 발생': '강도',
            '강간 발생': '강간',
            '절도 발생': '절도',
            '폭력 발생': '폭력'
        }, inplace=True)
        x = police[self.crime_rate_columns].values
        min_max_scalar = preprocessing.MinMaxScaler()
        '''
        스케일링은 선형변환을 적용하여
        전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정
        '''
        x_scaled = min_max_scalar.fit_transform(x.astype(float))
        '''
        정규화 normalization
        많은 양의 데이터를 처리함에 있어 데이터의 범위(도메인)를 일치시키거나
        분포(스케일)를 유사하게 만드는 작업
        '''
        police_norm = pd.DataFrame(x_scaled, columns=self.crime_columns, index=police.index)
        police_norm[self.crime_rate_columns] = police[self.crime_rate_columns]
        police_norm['범죄'] = np.sum(police_norm[self.crime_rate_columns], axis=1)
        police_norm['검거'] = np.sum(police_norm[self.crime_columns], axis=1)
        police_norm.to_pickle('save/crime/police_norm.pkl')
        print(pd.read_pickle('save/crime/police_norm.pkl'))

    def folium_example(self):
        us_unemployment = self.us_unemployment

        bins = list(us_unemployment['Unemployment'].quantile([0, 0.25, 0.5, 0.75, 1]))
        map = folium.Map(location=[48, -102], zoom_start=5)

        folium.Choropleth(
            geo_data=self.get_json_from_df("us-states.json"),  # us_states,
            data=us_unemployment,  # us_unemployment,
            name='choropleth',
            columns=['State', 'Unemployment'],
            key_on='feature.id',
            bins=bins,
            fill_color='YlGn',
            fill_opacity=0.7,
            line_opacity=0.5,
            legend_name='Unemployment Rate (%)',
            reset=True
        ).add_to(map)
        map.save('save/crime/Unemployment.html')

    def get_json_from_df(self):
        fname = 'data/crime/us-states.json'
        df = pd.read_json(fname)
        df.drop(df.index[[8, 51]], inplace=True)
        df.to_json(fname, orient='index')
        print(df)
        # return pd.read_json(fname)

    def save_seoul_folium(self):
        geo_data = self.kr_states
        data = self.creat_folium_data()

        map = folium.Map(location=[37.5502, 126.982], zoom_start=12)
        folium.Choropleth(
            geo_data=geo_data,  # us_states,
            data=data,  # us_unemployment,
            name='choropleth',
            columns=['State', 'Stime Rate'],
            key_on='feature.id',
            fill_color='PuRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Crime Rate (%)',
            reset=True
        ).add_to(map)
        map.save('save/crime/seoul_folium.html')

    def save_seoul_crime_map(self):  # 6
        mc = MyChoroplethVO()
        mc.geo_data = self.kr_states
        mc.data = self.get_seoul_crime_data()
        mc.name = "choropleth"
        mc.columns = ["State", "Crime Rate"]
        mc.key_on = "feature.id"
        mc.fill_color = "PuRd"
        mc.fill_opacity = 0.7
        mc.line_opacity = 0.2
        mc.legend_name = "Crime Rate (%)"
        mc.location = [37.5502, 126.982]
        mc.zoom_start = 12
        mc.save_path = "save/crime/seoul_crime_rate.html"
        MyChoroplethService(mc)

    def creat_folium_data(self):
        police_norm = pd.read_pickle('save/crime/police_norm.pkl')
        return tuple(zip(police_norm.index, police_norm['범죄']))


if __name__ == '__main__':
    def my_menu(ls):
        for i, j in enumerate(ls):
            print(f"{i}. {j}")
        return input('메뉴선택: ')


    t = CrimeSevrvice()
    while True:
        menu = my_menu(CRIME_MENUS)
        if menu == '0':
            print("종료")
            break
        else:
            crime_menu[menu](t)
