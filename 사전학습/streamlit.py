# 실행: streamlit run {your app}.py (port # : 8501)

import streamlit as st # 웹앱을 쉽게 만들 수 있는 library (pip install streamlit)
import pandas_datareader as pdr # pip install pandas-datareader

# 글자 출력 (markdown 문법 사용 가능)
st.write('''
# 삼성전자 주식 데이터
마감 가격과 거래량을 차트로 보여줍니다.
''')

df = pdr.get_data_yahoo('005930.KS', '2020-01-01', '2022-01-18')

st.line_chart(df.Close) # 선그래프
st.line_chart(df.Volume)