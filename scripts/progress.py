"""Small log-derived console status; never invent percentages or completion counts."""
import re
import time

LABELS = {
    "train": "训练", "selfplay-train": "训练自对弈", "selfplay-val": "验证自对弈",
    "selfplay-candidate-probe": "新模型试运行", "shuffle-train": "整理训练数据",
    "shuffle-val": "整理验证数据", "export": "导出权重", "evaluate": "新旧权重对战",
}


class Progress:
    def __init__(self, name, heartbeat=10):
        self.label = LABELS.get(name, name)
        self.started = time.monotonic()
        self.last_print = self.started
        self.last_status = None
        self.status = "正在启动、加载配置与模型"
        self.heartbeat = heartbeat

    def feed(self, line):
        if "BEGINNING NEXT EPOCH" in line:
            m = re.search(r"EPOCH\s+(\d+)", line)
            self.status = "进入 epoch " + m.group(1) if m else "进入下一 epoch"
        elif "Global step:" in line:
            m = re.search(r"Global step:\s*(\d+)", line)
            if m: self.status = f"已恢复训练状态，累计 {int(m.group(1)):,} 个样本"
        elif "Beginning training subepoch" in line:
            self.status = "正在训练：前向计算、反向传播和参数更新"
        elif "Finished training subepoch" in line:
            self.status = "本轮参数更新完成"
        elif "Beginning validation" in line:
            self.status = "正在计算验证集指标"
        elif "nsamp_train =" in line:
            m = re.search(r"nsamp_train =\s*([\d.]+)", line)
            if m: self.status = f"累计处理 {int(float(m.group(1))):,} 个训练样本"
        elif "Saving checkpoint:" in line:
            self.status = "正在保存 checkpoint"
        elif "Exceeding train bucket" in line:
            self.status = "新数据预算不足，正常结束，本次不再更新参数"
        elif "Hit max epochs this instance" in line:
            self.status = "已完成指定训练轮次"
        elif "Started " in line and " games with " in line:
            m = re.search(r"Started (\d+) games", line)
            if m: self.status = f"已启动 {m.group(1)} 局，正在对弈（包含尚未完成的棋局）"
        elif "Final games finished:" in line:
            m = re.search(r"Final games finished:\s*(\d+)", line)
            if m: self.status = f"已完成 {m.group(1)} 局，正在收尾保存数据"
        elif "Final data rows:" in line:
            m = re.search(r"Final data rows:\s*(\d+)", line)
            if m: self.status += f"；本次生成 {m.group(1)} 行数据"
        elif "Beginning:" in line:
            for token, text in [("Finding files", "扫描数据文件"), ("Computing rows", "统计数据行数"),
                                ("Sharding", "分片、打乱数据"), ("Merging", "合并并写入数据")]:
                if token in line: self.status = text
        elif "Writing model" in line:
            self.status = "正在写出引擎权重"
        elif "Loaded latest neural net" in line or "Loaded all config stuff" in line:
            self.status = "配置或模型已加载，正在准备对弈"

    def emit(self, force=False):
        now = time.monotonic()
        changed = self.status != self.last_status
        if force or now-self.last_print >= self.heartbeat or (changed and now-self.last_print >= 2):
            seconds = int(now-self.started)
            elapsed = f"{seconds//60:02d}:{seconds%60:02d}"
            print(f"[{self.label} {elapsed}] {self.status}", flush=True)
            self.last_print, self.last_status = now, self.status

    def finished(self, rc):
        elapsed = time.monotonic()-self.started
        print(f"[{self.label}] {'完成' if rc == 0 else '失败'}，用时 {elapsed:.1f} 秒。{self.status}", flush=True)
