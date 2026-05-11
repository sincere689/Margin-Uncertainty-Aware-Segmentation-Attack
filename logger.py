import os
import csv
import datetime


class ExperimentLogger:
    def __init__(self, args):
        """
        [升级版] 自动分类日志系统
        结构: experiments_log / 攻击算法名 / 时间_参数
        """
        self.args = args
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.root_dir = "experiments_log"

        # =================================================
        # 【核心改进】创建层级结构
        # =================================================

        # 1. 第一级：确保总目录存在
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # 2. 第二级：创建“算法专属”文件夹 (例如 experiments_log/SP-CoSPGD)
        # 这样不同算法的实验就会自动分家，不会混在一起
        self.attack_dir = os.path.join(self.root_dir, args.attack)
        if not os.path.exists(self.attack_dir):
            os.makedirs(self.attack_dir)

        # 3. 第三级：创建本次实验的时间戳文件夹 (例如 2026-01-21_..._eps0.031)
        folder_name = f"{self.timestamp}_eps{args.epsilon:.3f}"
        if args.debug:
            folder_name += "_DEBUG"  # 如果是调试模式，加个标记方便区分

        self.exp_dir = os.path.join(self.attack_dir, folder_name)
        os.makedirs(self.exp_dir)

        # =================================================

        # 准备文件路径
        self.config_file = os.path.join(self.exp_dir, "config.txt")
        self.summary_csv = os.path.join(self.root_dir, "all_results_summary.csv")  # 汇总表还在最外面

        # 执行记录
        self.log_config()
        self._init_csv_header()

    def log_config(self):
        """记录配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write("========== 实验配置单 ==========\n")
            f.write(f"执行时间: {datetime.datetime.now()}\n")

            # 基础实验信息
            f.write(f"攻击算法: {self.args.attack}\n")
            f.write(f"模型架构: {self.args.model_arch}\n")
            f.write(f"数据集: {self.args.dataset}\n")
            f.write(f"数据路径: {self.args.data_root}\n")

            # 攻击基本参数
            f.write(f"攻击强度(epsilon): {self.args.epsilon}\n")
            f.write(f"迭代次数(steps): {self.args.steps}\n")

            # 模块开关与关键超参数
            f.write(f"use_margin_weight: {self.args.effective_use_margin_weight}\n")
            f.write(f"use_uncertainty_blind_weight: {self.args.effective_use_uncertainty_blind_weight}\n")
            f.write(f"high_margin_ratio: {self.args.high_margin_ratio}\n")
            f.write(f"uncertainty_blind_lambda: {self.args.uncertainty_blind_lambda}\n")

            f.write("==============================\n")

        print(f"📁 [日志] 本次实验保存在: {self.exp_dir}")

    def _init_csv_header(self):
        """初始化汇总表"""
        if not os.path.exists(self.summary_csv):
            with open(self.summary_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Attack", "Epsilon", "Steps", "mIoU (%)", "Exp_Folder"])

    def log_result(self, attack_name, score):
        """写入结果"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 写汇总表
        with open(self.summary_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_time,
                attack_name,
                f"{self.args.epsilon:.4f}",
                self.args.steps,
                f"{score * 100:.2f}",
                self.exp_dir  # 这里记录的是分类后的深层路径
            ])

        # 写单次实验txt
        with open(self.config_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[最终结果] mIoU: {score:.4f} ({score * 100:.2f}%)")

        print(f"📝 [归档] 结果已存入 {attack_name} 分类文件夹。")