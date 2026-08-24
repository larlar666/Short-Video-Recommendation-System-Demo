@echo off
chcp 65001 > nul
echo ==============================
echo 1. Python路径
echo ==============================
where python
echo.
echo ==============================
echo 2. Python版本
echo ==============================
python --version
echo.
echo ==============================
echo 3. 安装核心依赖
echo ==============================
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo ==============================
echo 4. 单独确保 wordcloud 安装
echo ==============================
python -m pip install wordcloud --no-deps
echo.
echo ==============================
echo 5. 验证关键库
echo ==============================
python -c "import streamlit; print('streamlit OK:', streamlit.__version__)"
python -c "import wordcloud; print('wordcloud OK:', wordcloud.__version__)"
python -c "import pandas; print('pandas OK:', pandas.__version__)"
echo.
echo ==============================
echo 6. 启动 Streamlit
echo ==============================
echo 提示：首次启动会自动转换数据格式，约需额外30秒
python -m streamlit run streamlit_app.py
pause