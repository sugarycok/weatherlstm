from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from keras.models import load_model as keras_load_model
from datetime import timedelta
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Flask 애플리케이션 설정
app = Flask(__name__)

# 엑셀 파일 및 모델 파일 경로
default_excel_file_path = 'C:/Users/WSU/Desktop/Desktop (2) (2)/rnnpro4/예측결과.xlsx'
prediction_excel_file_path = 'C:/Users/WSU/Desktop/Desktop (2) (2)/rnnpro4/예측된_수정된_취합_일자_강수량_인구_유입량_20240829_제공.xlsx'
model_dir_path = 'C:/Users/WSU/Desktop/Desktop (2) (2)/rnnpro4/models/ALL_lstm_model.keras'

# 엑셀 파일 로드 및 데이터 준비 함수
def load_excel_data(file_path):
    excel_data = pd.ExcelFile(file_path)
    sheets_data = {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}
    print("엑셀 데이터 로드 완료.")  # 로그 출력
    return sheets_data

# 데이터 전처리 함수
def prepare_filtered_data(df):
    df.columns = df.columns.str.strip()

    # 날짜 컬럼 변환
    if '년월일' in df.columns:
        df['년월일'] = pd.to_datetime(df['년월일'], format='%Y-%m-%d %I:%M:%S %p', errors='coerce')

    # 수치형 데이터 변환
    numeric_cols = ['유입량(㎥/일)', '인구수(명)', '강수량(mm)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"{col}의 데이터: {df[col].unique()}")  # 데이터 확인

    # 결측치 처리
    for col in numeric_cols:
        if col in df.columns:
            if df[col].isna().sum() > 0:
                print(f"{col}에 결측치가 있습니다.")
                df[col].fillna(df[col].mean(), inplace=True)  # 평균으로 대체

    # 최근 1년치 데이터 필터링
    max_date = df['년월일'].max()
    one_year_ago = max_date - timedelta(days=365)
    df = df[df['년월일'] >= one_year_ago]

    print(f"데이터 전처리 완료: {len(df)}개의 데이터 포인트.")  # 로그 출력

    return df, one_year_ago

# 모델 불러오기 함수
def load_lstm_model():
    model_path = model_dir_path
    if os.path.exists(model_path):
        print("모델 로드 완료.")  # 로그 출력
        return keras_load_model(model_path)
    else:
        print("모델 파일이 존재하지 않습니다.")  # 로그 출력
        return None

# 예측 수행 함수
def make_prediction(df_filtered, model):
    input_scaler = MinMaxScaler()
    input_data = df_filtered[['인구수(명)', '강수량(mm)']].values
    
    # 데이터 스케일링
    scaled_input_data = input_scaler.fit_transform(input_data)
    preprocessed_data = scaled_input_data.reshape((scaled_input_data.shape[0], 1, scaled_input_data.shape[1]))

    predictions = model.predict(preprocessed_data)
    predictions = predictions.reshape(-1, 1)

    # 역스케일링
    predictions = input_scaler.inverse_transform(np.hstack([predictions, np.zeros((predictions.shape[0], 1))]))[:, 0]

    print(f"예측값 (스케일링 해제 후): {predictions[:10]}")  # 첫 10개 값 확인
    print(f"예측값 통계: min={predictions.min()}, max={predictions.max()}, mean={predictions.mean()}")  # 예측값 통계 출력

    return predictions

# 예측 결과를 엑셀 파일에 저장하는 함수
def save_predictions_to_excel(predictions, df_filtered, file_path):
    df_filtered.loc[:, '예측 유입량(㎥/일)'] = predictions  # .loc를 사용하여 경고 방지
    
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        existing_sheets = writer.book.sheetnames
        if '예측 결과' in existing_sheets:
            del writer.book['예측 결과']  # 기존 시트 삭제
        df_filtered.to_excel(writer, sheet_name='예측 결과', index=False)
    
    print("예측 결과를 엑셀 파일에 저장했습니다.")

# 그래프 생성 함수
def create_graph(df_filtered, selected_sheet, one_year_ago):
    fig = go.Figure()

    # 실제 유입량 그래프 (천 단위로 변환)
    fig.add_trace(go.Scatter(
        x=df_filtered['년월일'], 
        y=df_filtered['유입량(㎥/일)'] / 1000,  # 천 단위로 변환
        mode='lines',  
        name='실제 유입량 (천 단위)',
        line=dict(color='blue', width=2),
        connectgaps=False
    ))

    # 예측 유입량 그래프 (천 단위로 변환)
    if '예측 유입량(㎥/일)' in df_filtered.columns:
        predicted_data = df_filtered[['년월일', '예측 유입량(㎥/일)']]
        predicted_data = predicted_data[predicted_data['년월일'] >= one_year_ago]
        
        if not predicted_data.empty:
            fig.add_trace(go.Scatter(
                x=predicted_data['년월일'], 
                y=predicted_data['예측 유입량(㎥/일)'] / 1000,  # 천 단위로 변환
                mode='lines',  
                name='예측 유입량 (천 단위)',
                line=dict(color='red', width=2),  # 실선으로 설정
                connectgaps=False
            ))

    # y축 범위 설정
    min_y = min(df_filtered['유입량(㎥/일)'].min() / 1000, predicted_data['예측 유입량(㎥/일)'].min() / 1000)
    max_y = max(df_filtered['유입량(㎥/일)'].max() / 1000, predicted_data['예측 유입량(㎥/일)'].max() / 1000)
    
    fig.update_yaxes(range=[min_y * 0.9, max_y * 1.1])

    # 인구수 그래프 (두 번째 y축)
    fig.add_trace(go.Scatter(
        x=df_filtered['년월일'], 
        y=df_filtered['인구수(명)'], 
        mode='lines',  
        name='인구수',
        line=dict(color='green', width=2),
        connectgaps=False,
        yaxis='y2'
    ))

    fig.update_layout(
        title=f'{selected_sheet} 데이터 그래프',
        xaxis_title='날짜',
        yaxis_title='유입량 (천 단위)',
        yaxis2=dict(title='인구수(명)', overlaying='y', side='right')
    )

    print("그래프 생성 완료.")
    return fig

# Flask 루트 경로
@app.route('/', methods=['GET', 'POST'])
def index():
    sheets_data = load_excel_data(default_excel_file_path)
    selected_sheet = request.form.get('sheet', '가좌')

    if request.method == 'POST' and 'reset' in request.form:
        return redirect(url_for('index'))

    population_growth = float(request.form.get('populationGrowth', 0))
    precipitation_status = request.form.get('precipitationStatus', 'normal')
    graph_type = request.form.get('graphType', 'daily')

    df_filtered, one_year_ago = prepare_filtered_data(sheets_data[selected_sheet])
    fig = create_graph(df_filtered, selected_sheet, one_year_ago)
    graph_html = pio.to_html(fig, full_html=False)

    temp_data = df_filtered.copy()
    modified_population = None
    modified_precipitation = None

    if request.method == 'POST' and (population_growth != 0 or precipitation_status != 'normal'):
        if population_growth != 0:
            original_population = temp_data['인구수(명)'].iloc[-1]
            # 인구 수 조정: -50%에서 +50% 사이의 값 적용
            adjustment_factor = 1 + (population_growth / 100)  # 예: +10%는 1.1, -10%는 0.9
            temp_data.loc[:, '인구수(명)'] *= adjustment_factor
            
            modified_population = temp_data['인구수(명)'].iloc[-1]
            print(f"원래 인구 수: {original_population}, 수정된 인구 수: {modified_population}")

        # 강수량 상태에 따른 조정
        if precipitation_status == 'drought':
            temp_data.loc[:, '강수량(mm)'] *= 0.5
        elif precipitation_status == 'flood':
            temp_data.loc[:, '강수량(mm)'] *= 1.5

        # 예측을 위한 데이터 로드
        prediction_data = load_excel_data(prediction_excel_file_path)[selected_sheet]
        prediction_df_filtered, _ = prepare_filtered_data(prediction_data)

        # LSTM 모델을 사용하여 예측 수행
        model = load_lstm_model()
        if model is not None:
            predictions = make_prediction(prediction_df_filtered, model)
            df_filtered.loc[:, '예측 유입량(㎥/일)'] = predictions

            # 예측 결과를 엑셀 파일에 저장
            save_predictions_to_excel(predictions, df_filtered, prediction_excel_file_path)
        else:
            print("모델을 불러오는 데 실패했습니다.")

        # 수정된 데이터로 그래프 다시 생성
        fig = create_graph(df_filtered, selected_sheet, one_year_ago)
        graph_html = pio.to_html(fig, full_html=False)

    # 예측 데이터 표 생성
    predictions_df = df_filtered[['년월일', '예측 유입량(㎥/일)']].dropna()
    recent_30_days = df_filtered['년월일'].max() - pd.DateOffset(days=30)
    predictions_df = predictions_df[predictions_df['년월일'] >= recent_30_days]

    # 열 이름 변경
    table_html = predictions_df.rename(columns={'년월일': '날짜', '예측 유입량(㎥/일)': '예측 유입량(㎥/일)'}).to_html(classes='table table-striped table-bordered table-center', index=False)

    return render_template('index.html',
                           graph_html=graph_html,
                           table_html=table_html,
                           selected_sheet=selected_sheet,
                           sheets=sheets_data.keys(),
                           graph_type=graph_type,
                           population_growth=population_growth,  # 원래 슬라이더 값을 반환하기 위해
                           precipitation_status=precipitation_status,
                           modified_population=modified_population,
                           modified_precipitation=modified_precipitation)

if __name__ == '__main__':
    app.run(debug=True)
