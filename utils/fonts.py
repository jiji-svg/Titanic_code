import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def fonting():
    # 한글 폰트 경로 설정 (윈도우 기본 맑은 고딕 예시)
    font_path = 'C:/Windows/Fonts/malgun.ttf'  
    font_name = fm.FontProperties(fname=font_path).get_name()

    plt.rc('font', family=font_name)  # 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지