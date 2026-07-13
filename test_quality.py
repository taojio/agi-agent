from agi_agent.file_ingestion.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

test_texts = [
    "SH#600000\nThis is a test file for stock data analysis\nStock: Shanghai Stock Exchange\nCode: 600000\nPrice: 12.50\nVolume: 1000000",
    "SH#600000",
    "SH600000",
    "600