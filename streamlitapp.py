#pip install poetry
##need to run poetry init to create pyproject.toml file
#poetry add pandasai

#pip install streamlit

#pip install pip install numpy==1.26.4
#pip install pandasai[seaborn]


import matplotlib
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.connectors import PandasConnector
from pandasai.connectors.yahoo_finance import YahooFinanceConnector
from pandasai.llm import BambooLLM
from pandasai.responses.response_parser import ResponseParser
import os
os.environ['PANDASAI_API_KEY'] = '$2a$10$zYXEMaIw0fLSjGSq3MrVDuNV9HQ85BfV4O5s5fxSBoCrediuGQcg2'



class OutputParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
    def parse(self, result):
        if result['type'] == "dataframe":
            st.dataframe(result['value'])
        elif result['type'] == 'plot':
            st.image(result["value"])
        else:
            st.write(result['value'])
        return


def setup():
    st.header("Chat with your small and large datasets!", anchor=False, divider="red")

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


def get_tasks():
    st.sidebar.header("Select a task", divider="rainbow")
    task = st.sidebar.radio("Choose one:",
                            ("Load from local drive, <200MB",
                             "Load from local drive, 200MB+",
                             "Yahoo Finance")
                            )
    return task


def main():
    """1. Setup page
       2. Setup options - tasks: load or retrieve, model: BambooLLM
       3. Yahoo Finance
    """
    setup()
    task = get_tasks()

    if task == "Load from local drive, <200MB":
        dataset = st.file_uploader("Upload your CSV or XLSX file", type=['csv', 'xlsx'])
        if not dataset:
            st.stop()
        df = pd.read_csv(dataset, low_memory=False)
        st.write("Data Preview:")
        st.dataframe(df.head())
        col_desc = st.radio("Do you want to provide column descriptors?",
                            ("Yes",
                             "No")
                            )
        if col_desc == "Yes":
            addon = st.text_input("Enter your column description, e.g. 'col1': 'unique id'")
        else:
            addon = "None"

        if addon:
            llm = BambooLLM()
            connector = PandasConnector({"original_df": df}, field_descriptions=addon)
            sdf = SmartDataframe(connector, {"enable_cache": False}, config={
                "llm": llm,
                "conversational": False,
                "response_parser": OutputParser
            })
            prompt = st.text_input("Enter your question/prompt.")
            if not prompt:
                st.stop()
            st.write("Response")
            response = sdf.chat(prompt)
            st.divider()
            st.write("🧞‍♂️ Under the hood, the code that was executed:")
            st.code(sdf.last_code_executed)

    if task == "Load from local drive, 200MB+":
        filename = st.text_input("Enter your file path including filename, e.g. /users/xyz/abc.csv (CSV files only)")
        if not filename:
            st.stop()
        df_large = pd.read_csv(filename, low_memory=False)
        st.write("Data Preview:")
        st.dataframe(df_large.head())
        col_desc = st.radio("Do you want to provide column descriptors?",
                            ("Yes",
                             "No")
                            )
        if col_desc == "Yes":
            addon = st.text_input("Enter your column description, e.g. 'col1': 'unique id'")
        else:
            addon = "None"

        if addon:
            llm = BambooLLM()
            connector = PandasConnector({"original_df": df_large}, field_descriptions=addon)
            sdf = SmartDataframe(connector, {"enable_cache": False}, config={
                "llm": llm,
                "conversational": False,
                "response_parser": OutputParser
            })
            prompt = st.text_input("Enter your question/prompt.")
            if not prompt:
                st.stop()
            st.write("Response")
            response = sdf.chat(prompt)
            st.divider()
            st.write("🧞‍♂️ Under the hood, the code that was executed:")
            st.code(sdf.last_code_executed)

    if task == "Yahoo Finance":
        stock_symbol = st.text_input("Enter a stock symbol, e.g. MSFT.")
        if not stock_symbol:
            st.stop()
        yahoo_connector = YahooFinanceConnector(stock_symbol)
        llm = BambooLLM()
        yahoo_df = SmartDataframe(yahoo_connector, config={"llm": llm, "response_parser": OutputParser})
        prompt = st.text_input("Enter your prompt.")
        if not prompt:
            st.stop()
        st.write("Response")
        response = yahoo_df.chat(prompt)
        st.divider()
        st.code(yahoo_df.last_code_executed)


if __name__ == '__main__':
    matplotlib.use("Agg", force=True)
    main()
