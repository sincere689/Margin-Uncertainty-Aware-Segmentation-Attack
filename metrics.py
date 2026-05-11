import numpy as np
import os

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes))

    def update(self, a, b):
        a = a.flatten()
        b = b.flatten()
        k = (a >= 0) & (a < self.num_classes)
        inds = self.num_classes * a[k].astype(int) + b[k].astype(int)
        self.mat += np.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def reset(self):
        self.mat.fill(0)

    def get_results(self):
        h = self.mat
        # mIoU
        iu = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h) + 1e-10)
        mean_iou = np.nanmean(iu)
        # mAcc
        class_acc = np.diag(h) / (h.sum(axis=1) + 1e-10)
        mean_acc = np.nanmean(class_acc)
        return {
            "Mean IoU": mean_iou,
            "Mean Acc": mean_acc
        }

    # 【新增】保存矩阵到硬盘的方法
    def save_to_file(self, folder, attack_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f"confusion_matrix_{attack_name}.npy")
        np.save(filename, self.mat)
        print(f"💾 [备份] 混淆矩阵已保存至: {filename}")