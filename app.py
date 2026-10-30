import os
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

MODEL_PATH = "mnist_model.pt"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = DigitCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    except Exception:
        model = None

HISTORY_FILE = "prediction_history.csv"

def init_history_file():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Predicted Digit", "Confidence"])

def predict_digit(pil_image):
    if model is None:
        return "No Model", "Run train.py first"

    try:
        img_gray = pil_image.convert("L")
        img_array = np.array(img_gray)
        
        # Locate non-zero pixel bounds
        rows = np.any(img_array > 20, axis=1)
        cols = np.any(img_array > 20, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            img_processed = Image.fromarray(img_array).resize((28, 28), Image.Resampling.LANCZOS)
        else:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            cropped = img_gray.crop((xmin, ymin, xmax + 1, ymax + 1))
            
            # PRESERVE ASPECT RATIO: Create square canvas with padding
            w, h = cropped.size
            max_side = max(w, h)
            
            # Add 20% border around digit
            padding = int(max_side * 0.2)
            square_size = max_side + (padding * 2)
            
            square_img = Image.new("L", (square_size, square_size), 0)
            offset_x = (square_size - w) // 2
            offset_y = (square_size - h) // 2
            square_img.paste(cropped, (offset_x, offset_y))
            
            img_processed = square_img.resize((28, 28), Image.Resampling.LANCZOS)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tensor_img = transform(img_processed).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return int(predicted.item()), f"{float(confidence.item()) * 100:.2f}%"
    except Exception as e:
        return "Error", f"Failed: {str(e)}"

class SingleDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("760x450")
        self.root.resizable(False, False)
        
        init_history_file()
        
        self.canvas_size = 250
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.pil_image)
        self.tk_photo = None
        
        self.setup_ui()
        self.load_history()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        ttk.Label(left_frame, text="Draw Single Digit (0-9) or Upload Image:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=1, highlightbackground="gray")
        self.canvas.pack(anchor=tk.W)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 5))
        
        ttk.Button(btn_frame, text="🔮 Predict", command=self.process_prediction).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="📁 Upload", command=self.upload_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="🧹 Clear", command=self.clear_canvas).pack(side=tk.LEFT)
        
        self.result_label = ttk.Label(left_frame, text="Predicted Digit: --", font=("Arial", 13, "bold"))
        self.result_label.pack(anchor=tk.W, pady=(8, 2))
        
        self.confidence_label = ttk.Label(left_frame, text="Confidence: --", font=("Arial", 10))
        self.confidence_label.pack(anchor=tk.W)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        hdr = ttk.Frame(right_frame)
        hdr.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(hdr, text="📊 History", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(hdr, text="Delete Logs", command=self.clear_history).pack(side=tk.RIGHT)
        
        columns = ("datetime", "digit", "confidence")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings", height=14)
        self.tree.heading("datetime", text="Timestamp")
        self.tree.heading("digit", text="Digit")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("datetime", width=140, anchor=tk.CENTER)
        self.tree.column("digit", width=60, anchor=tk.CENTER)
        self.tree.column("confidence", width=90, anchor=tk.CENTER)
        
        sb = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def paint(self, event):
        r = 8
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            try:
                uploaded = Image.open(file_path).convert("L")
                if np.mean(np.array(uploaded.resize((10, 10)))) > 127:
                    uploaded = ImageOps.invert(uploaded)
                
                uploaded = uploaded.resize((self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS)
                self.pil_image = uploaded
                self.draw = ImageDraw.Draw(self.pil_image)
                
                self.tk_photo = ImageTk.PhotoImage(uploaded)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_photo)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.pil_image)
        self.tk_photo = None
        self.result_label.config(text="Predicted Digit: --")
        self.confidence_label.config(text="Confidence: --")

    def process_prediction(self):
        digit, conf = predict_digit(self.pil_image)
        self.result_label.config(text=f"Predicted Digit: {digit}")
        self.confidence_label.config(text=f"Confidence: {conf}")
        
        if str(digit) not in ["Error", "No Model"]:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([now, digit, conf])
            self.load_history()

    def load_history(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reversed(list(reader)):
                    if row:
                        self.tree.insert("", tk.END, values=row)

    def clear_history(self):
        if messagebox.askyesno("Confirm", "Clear prediction log history?"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            init_history_file()
            self.load_history()

if __name__ == "__main__":
    root = tk.Tk()
    app = SingleDigitApp(root)
    root.mainloop()