import joblib
import streamlit as st
import numpy as np
import pandas as pd


SVMmodel = joblib.load('sphalerite7.4/Dependencies/RF_Sphalerite Classifier.pkl')
RFmodel = joblib.load('sphalerite7.4/Dependencies/RF_Sphalerite Classifier.pkl')
feature_names = ['Cd', 'Mn', 'Ag', 'Cu', 'Pb', 'Sn', 'Ga', 'In', 'Sb', 'Co', 'Ge', 'Fe']

prediction_labels = {
    0: 'MVT',
    1: 'Sedex',
    2: 'Skarn',
    3: 'VMS'
}

# 主应用程序
def main():
    st.title('闪锌矿微量元素的矿床分类预测模型')

    # model select
    model_select = st.selectbox(
        "模型选择",
        ['SVMmodel', 'RFmodel']
    )
    # 创建输入框来接收特征值
    feature_inputs = []
    for i in range(len(feature_names)):
        feature = st.number_input(f'输入{feature_names[i]}的值', value=0.00001)
        feature_inputs.append(feature)
    # 将特征转换为 NumPy 数组
    features = np.array(feature_inputs).reshape(1, -1)
    # 检查输入数据中是否有零或负数值
    if np.any(features <= 0):
        st.warning('输入数据必须为正数值，请重新输入有效的特征值。')
        return
    # 显示预测结果
    if st.button('预测'):
        if model_select == 'SVMmodel':
            prediction = SVMmodel.predict(features)
        else:
            prediction = RFmodel.predict(features)
        st.subheader('预测结果')
        st.write(f'预测结果：{prediction_labels[prediction[0]]}')

if __name__ == '__main__':
    main()
