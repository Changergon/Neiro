import tkinter as tk
from tkinter import filedialog

from cluster_and_expand import run_clustering


class ClusterGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Графический интерфейс кластеризации")
        self.master.geometry("400x300")

        self.unknown_dir = ""
        self.cluster_dir = ""
        self.new_class_dir = ""
        self.model_path = ""
        self.num_classes = 0
        self.batch_size = 64
        self.threshold = 0.8
        self.eps = 0.5
        self.min_samples = 5

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Выберите папку с неизвестными изображениями").pack(pady=10)
        tk.Button(self.master, text="Выбрать папку", command=self.select_unknown_dir).pack(pady=5)

        tk.Label(self.master, text="Выберите папку для кластеров").pack(pady=10)
        tk.Button(self.master, text="Выбрать папку", command=self.select_cluster_dir).pack(pady=5)

        tk.Label(self.master, text="Выберите папку для новых классов").pack(pady=10)
        tk.Button(self.master, text="Выбрать папку", command=self.select_new_class_dir).pack(pady=5)

        tk.Label(self.master, text="Выберите файл модели").pack(pady=10)
        tk.Button(self.master, text="Выбрать модель", command=self.select_model_file).pack(pady=5)

        tk.Button(self.master, text="Начать кластеризацию", command=self.start_clustering).pack(pady=20)

    def select_unknown_dir(self):
        self.unknown_dir = filedialog.askdirectory()

    def select_cluster_dir(self):
        self.cluster_dir = filedialog.askdirectory()

    def select_new_class_dir(self):
        self.new_class_dir = filedialog.askdirectory()

    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])

    def start_clustering(self):
        if not (self.unknown_dir and self.cluster_dir and self.new_class_dir and self.model_path):
            print("Пожалуйста, выберите все необходимые папки и модель.")
            return

        run_clustering(self.unknown_dir, self.cluster_dir, self.new_class_dir,
                       self.model_path, self.num_classes, self.threshold,
                       self.batch_size, self.eps, self.min_samples)


if __name__ == "__main__":
    root = tk.Tk()
    app = ClusterGUI(root)
    root.mainloop()
