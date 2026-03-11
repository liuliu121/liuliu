import os
import glob
import numpy as np
import pandas as pd

# ================= 配置区域 =================
# 请确保这里的路径指向你的实验结果目录
# 根据你的日志 checkpoints/CIF_MMIN_MOSI_block_5_run_0_1 推测，日志目录应该是 logs/CIF_MMIN_MOSI
# 如果你的 name 参数不同，请修改这里。例如: './logs/CIF_MMIN_IEMOCAP/results'
RESULTS_DIR = './logs/CIF_MMIN_MOSI_block_5_run_0_1/results'
# ===========================================

def load_and_calculate_mean(file_path):
    try:
        # 读取 TSV 文件
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 过滤有效行：
        # 1. 跳过表头 (第一行通常是 acc uar f1，但 MOSI 实际存的是 mae corr f1)
        # 2. 跳过空行
        # 3. 提取 10 折的数据 (假设只有 10 行数据是有效的 CV 结果)
        
        data_rows = []
        for line in lines[1:]: # 跳过第一行表头
            parts = line.strip().split('\t')
            if len(parts) == 3: # 确保这一行有 3 个数据
                try:
                    # 尝试转换为 float
                    vals = [float(p) for p in parts]
                    data_rows.append(vals)
                except ValueError:
                    continue

        if len(data_rows) == 0:
            print(f"Warning: No valid data found in {os.path.basename(file_path)}")
            return None

        # 转换为 numpy 数组方便计算
        data_np = np.array(data_rows)
        
        # 如果数据行数超过 10 行，可能包含了之前自动写入的平均值行，我们需要取前 10 行（1-10折）
        if data_np.shape[0] > 10:
             data_np = data_np[:10]
             
        # 计算均值
        means = np.mean(data_np, axis=0)
        return means, data_np.shape[0]

    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory '{RESULTS_DIR}' does not exist.")
        print("Please check the 'RESULTS_DIR' variable in the script.")
        return

    # 获取目录下所有 tsv 文件
    tsv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'result_*.tsv')))
    
    if not tsv_files:
        print("No .tsv result files found.")
        return

    print(f"{'Condition':<15} | {'MAE':<10} | {'Corr':<10} | {'F1':<10} | {'Folds':<5}")
    print("-" * 65)

    # 这里的列名对应代码中写入的顺序。
    # 对于 MOSI，train_miss.py 写入顺序是: mae, corr, f1
    # 对于 IEMOCAP/MSP，写入顺序是: acc, uar, f1
    # 根据你的日志输出 "Tst result mae ... corr ... f1 ...", 这是一个 MOSI 任务。
    
    results_summary = []

    for file_path in tsv_files:
        file_name = os.path.basename(file_path)
        # 提取条件名称，例如 result_total.tsv -> total, result_azz.tsv -> azz
        condition = file_name.replace('result_', '').replace('.tsv', '')
        
        res = load_and_calculate_mean(file_path)
        if res:
            means, num_folds = res
            # MOSI: col 0 = MAE, col 1 = Corr, col 2 = F1
            print(f"{condition:<15} | {means[0]:<10.4f} | {means[1]:<10.4f} | {means[2]:<10.4f} | {num_folds:<5}")
            results_summary.append((condition, means))

    print("-" * 65)
    print("Note: For MOSI dataset, columns are typically MAE, Corr, F1.")
    print("      For IEMOCAP/MSP, columns are typically Acc, UAR, F1.")
    print(f"Read from: {RESULTS_DIR}")

if __name__ == "__main__":
    main()