import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import logging
from openpyxl.styles import Alignment

# 로그 설정
logging.basicConfig(level=logging.INFO)

def predict_missing_values(df):
    # 년월일을 년, 월, 일로 분리하여 추가
    df['년'] = pd.to_datetime(df['년월일']).dt.year
    df['월'] = pd.to_datetime(df['년월일']).dt.month
    df['일'] = pd.to_datetime(df['년월일']).dt.day

    # 열 이름 정리: 줄바꿈 문자 제거
    df.columns = df.columns.str.replace('\n', '')

    # 모델 학습을 위한 피처와 타겟 설정
    features = ['강수량(mm)', '인구수(명)', '년', '월', '일']
    target = '유입량(㎥/일)'

    # 유입량 열이 있는지 확인
    if target not in df.columns:
        logging.warning(f"'{target}' column not found in the data.")
        return df

    # 결측값이 없는 데이터를 사용해 모델을 학습
    train_data = df.dropna(subset=[target])
    X_train = train_data[features]
    y_train = train_data[target]

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 모델 최적화 및 평가
    best_model = optimize_and_evaluate_model(X_train_scaled, y_train)

    # 결측값이 있는 데이터를 예측
    missing_data = df[df[target].isna()]
    if not missing_data.empty:
        X_missing = missing_data[features]
        X_missing_scaled = scaler.transform(X_missing)
        predicted_values = best_model.predict(X_missing_scaled)

        # 예측된 값을 비어 있는 곳에만 채움
        df.loc[df[target].isna(), target] = predicted_values

    return df

def optimize_and_evaluate_model(X_train, y_train):
    # 하이퍼파라미터 그리드 정의
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best score: {grid_search.best_score_}")

    return grid_search.best_estimator_

def adjust_column_width_and_alignment(excel_writer, sheet_name):
    workbook = excel_writer.book
    worksheet = workbook[sheet_name]
    
    for col in worksheet.columns:
        max_length = 0
        column = col[0].column_letter  # Get the column name
        for cell in col:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
            
            # 가운데 정렬 적용
            cell.alignment = Alignment(horizontal='center')
        
        # 여유 공간을 양쪽에 균등하게 배분
        adjusted_width = (max_length + 4) * 1.1  # 패딩을 추가하여 양쪽에 여유 공간 확보
        worksheet.column_dimensions[column].width = adjusted_width

def process_sheets(file_path, output_path):
    # 엑셀 파일 불러오기
    xlsx = pd.ExcelFile(file_path)

    # 모든 시트에 대해 처리
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)

            # '강수량(mm)' 열의 빈 값을 0으로 채우기
            if '강수량(mm)' in df.columns:
                df['강수량(mm)'] = df['강수량(mm)'].fillna(0)

            # '인구수(명)' 열의 빈 값을 0으로 채우기
            if '인구수(명)' in df.columns:
                df['인구수(명)'] = df['인구수(명)'].fillna(0)

            # 유입량 열의 결측값만 예측하여 채우기
            logging.info(f"Predicting missing values in '{sheet_name}'")
            df = predict_missing_values(df)

            # '년', '월', '일' 열 제거
            df = df.drop(columns=['년', '월', '일'])

            # 처리된 데이터를 새로운 엑셀 파일에 저장
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            logging.info(f"Saved processed sheet: {sheet_name}")

            # 열 크기 자동 조정 및 가운데 정렬
            adjust_column_width_and_alignment(writer, sheet_name)

# 파일 경로 설정
file_path = '취합_일자_강수량_인구_유입량_20240829_제공.xlsx'
output_path = '예측된_수정된_취합_일자_강수량_인구_유입량_20240829_제공.xlsx'

# 모든 탭에 대해 처리 실행
process_sheets(file_path, output_path)
