import argparse
import pandas as pd


def merge_xlsx(file1: str, file2: str, output: str):
    """合并两个 xlsx 文件，纵向拼接所有 sheet。"""
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    print(f"File 1: {file1}")
    print(f"  shape: {df1.shape}, columns: {list(df1.columns)}")
    print(f"File 2: {file2}")
    print(f"  shape: {df2.shape}, columns: {list(df2.columns)}")

    merged = pd.concat([df1, df2], ignore_index=True)
    print(f"\nMerged shape: {merged.shape}")

    merged.to_excel(output, index=False)
    print(f"[✓] Saved to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge two xlsx files.")
    parser.add_argument("--file1", type=str, required=True, help="第一个 xlsx 文件路径")
    parser.add_argument("--file2", type=str, required=True, help="第二个 xlsx 文件路径")
    parser.add_argument("--output", type=str, required=True, help="合并后输出的 xlsx 文件路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_xlsx(args.file1, args.file2, args.output)
