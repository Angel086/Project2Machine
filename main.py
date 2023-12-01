import tensorflow
import getData

def process_data(df):
    target = df.pop('Close/Last')
    print(target)

def main():
    df = getData.read_data()
    process_data(df)
    print("hello world")

if __name__ == "__main__":
    main();