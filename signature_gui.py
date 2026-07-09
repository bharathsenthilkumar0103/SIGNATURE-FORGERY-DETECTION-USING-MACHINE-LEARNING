import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim
import threading
import time

# ==========================================================
#                  ADVANCED PREPROCESSING FUNCTION
# ==========================================================
def preprocess_image(img_path):
    """
    Preprocessing for color and noise normalization so that
    signatures with different ink colors or lighting still compare correctly.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Unable to read image.")

    # Resize for uniformity
    img = cv2.resize(img, (300, 150))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize lighting using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoising
    gray = cv2.fastNlMeansDenoising(gray, None, h=20)

    # Adaptive thresholding (remove lighting bias)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )

    # Morphological cleaning
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Invert to keep black signature on white background
    cleaned = cv2.bitwise_not(binary)

    # Normalize to [0, 1]
    return cleaned / 255.0


# ==========================================================
#                FEATURE + SIMILARITY COMPUTATION
# ==========================================================
def compute_similarity(img1_path, img2_path, debug=False):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    # Structural similarity
    ssim_score, _ = ssim(img1, img2, full=True, data_range=1.0)

    # Euclidean distance
    euclid = np.linalg.norm(img1.flatten() - img2.flatten())

    # ORB keypoints
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute((img1 * 255).astype(np.uint8), None)
    kp2, des2 = orb.detectAndCompute((img2 * 255).astype(np.uint8), None)
    orb_score = 0
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if matches:
            orb_score = len(matches) / max(len(kp1), len(kp2))

    # Edge-based difference
    edges1 = cv2.Canny((img1 * 255).astype(np.uint8), 50, 150)
    edges2 = cv2.Canny((img2 * 255).astype(np.uint8), 50, 150)
    edge_diff = np.sum(np.abs(edges1 - edges2)) / 255
    edge_diff_ratio = edge_diff / edges1.size

    if debug:
        print(f"[METRICS] SSIM={ssim_score:.3f}, ORB={orb_score:.3f}, EDGE_DIFF={edge_diff_ratio:.3f}, EUCLID={euclid:.1f}")

    # Smart forged rejection
    low_ssim = ssim_score < 0.55
    bad_edges = edge_diff_ratio > 0.25
    high_euclid = euclid > 35.0
    weak_orb = orb_score < 0.15

    if ((low_ssim and bad_edges) or (low_ssim and high_euclid) or (weak_orb and low_ssim)):
        if debug:
            print("[DECISION] Strong forged rejection triggered.")
        return 0.0

    # Weighted combination
    combined = (0.6 * ssim_score) + (0.25 * orb_score) + (0.15 * (1 - edge_diff_ratio))
    combined = np.clip(combined, 0, 1)

    # Nonlinear mapping to percentage
    if combined >= 0.95:
        similarity = 100.0
    elif combined >= 0.80:
        similarity = 80 + (combined - 0.80) * 100
    elif combined >= 0.65:
        similarity = 60 + (combined - 0.65) * 133
    elif combined >= 0.50:
        similarity = 45 + (combined - 0.50) * 100
    else:
        similarity = combined * 40

    return float(np.clip(similarity, 0, 100))


# ==========================================================
#                        GUI DESIGN
# ==========================================================
class SignatureChecker:
    def __init__(self, root):
        self.root = root
        self.root.title("🖋 Signature Similarity Checker")
        self.root.geometry("520x780")
        self.root.resizable(False, False)

        # Background gradient
        self.bg = tk.Canvas(root, width=520, height=780)
        self.bg.pack(fill="both", expand=True)
        self.draw_gradient(self.bg, "#74ebd5", "#ACB6E5")

        Label(root, text="Signature Similarity Checker",
              font=("Segoe UI", 18, "bold"), bg="#ACB6E5", fg="#1e272e").place(x=70, y=25)

        self.canvas1 = Label(root, bg="#dff9fb", width=300, height=150, relief="groove", bd=2)
        self.canvas1.place(x=110, y=80)
        self.canvas2 = Label(root, bg="#dff9fb", width=300, height=150, relief="groove", bd=2)
        self.canvas2.place(x=110, y=270)

        self.make_button("Upload Signature 1", 120, 240, self.upload_sig1)
        self.make_button("Upload Signature 2", 120, 430, self.upload_sig2)
        self.make_button("🧹 Clean & Preview Signatures", 140, 500, self.clean_and_preview, "#00b894")
        self.make_button("Compare Signatures", 140, 550, self.compare_signatures, "#1dd1a1")

        self.result_label = Label(root, text="", font=("Segoe UI", 14, "bold"), bg="#ACB6E5")
        self.result_label.place(x=100, y=610)

        self.accuracy_bar = tk.Canvas(root, width=300, height=30, bg="#dcdde1", highlightthickness=0)
        self.accuracy_bar.place(x=110, y=670)

        self.sig1_path = None
        self.sig2_path = None

    def draw_gradient(self, canvas, color1, color2):
        (r1, g1, b1) = self.root.winfo_rgb(color1)
        (r2, g2, b2) = self.root.winfo_rgb(color2)
        r_ratio = (r2 - r1) / 780
        g_ratio = (g2 - g1) / 780
        b_ratio = (b2 - b1) / 780
        for i in range(780):
            nr = int(r1 + (r_ratio * i))
            ng = int(g1 + (g_ratio * i))
            nb = int(b1 + (b_ratio * i))
            color = f"#{nr>>8:02x}{ng>>8:02x}{nb>>8:02x}"
            canvas.create_line(0, i, 520, i, fill=color)

    def make_button(self, text, x, y, cmd, bg="#0984e3"):
        Button(self.root, text=text, command=cmd, font=("Segoe UI", 10, "bold"),
               bg=bg, fg="white", activebackground="#74b9ff", relief="flat",
               width=25, height=1, cursor="hand2").place(x=x, y=y)

    def upload_sig1(self):
        self.sig1_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if self.sig1_path:
            self.display_image(self.sig1_path, self.canvas1)

    def upload_sig2(self):
        self.sig2_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if self.sig2_path:
            self.display_image(self.sig2_path, self.canvas2)

    def display_image(self, path, label):
        img = Image.open(path).resize((300, 150))
        label.img = ImageTk.PhotoImage(img)
        label.config(image=label.img)

    def clean_and_preview(self):
        if not self.sig1_path or not self.sig2_path:
            self.result_label.config(text="⚠ Please upload both signatures first!", fg="red")
            return

        img1 = (preprocess_image(self.sig1_path) * 255).astype(np.uint8)
        img2 = (preprocess_image(self.sig2_path) * 255).astype(np.uint8)

        cv2.imwrite("cleaned_sig1.png", img1)
        cv2.imwrite("cleaned_sig2.png", img2)

        self.display_image("cleaned_sig1.png", self.canvas1)
        self.display_image("cleaned_sig2.png", self.canvas2)
        self.result_label.config(text="🧼 Cleaned & Normalized for Comparison", fg="#00b894")

    def compare_signatures(self):
        if not self.sig1_path or not self.sig2_path:
            self.result_label.config(text="⚠ Please upload both signatures!", fg="red")
            return

        acc = compute_similarity(self.sig1_path, self.sig2_path, debug=False)

        if acc == 100:
            verdict, color = "✅ IDENTICAL SIGNATURE", "#27ae60"
        elif acc >= 60:
            verdict, color = "✍️ SIMILAR (Same Signer)", "#2ecc71"
        else:
            verdict, color = "❌ FORGED SIGNATURE", "#e74c3c"

        self.result_label.config(text=f"{verdict}\nSimilarity: {acc:.2f}%", fg=color)
        threading.Thread(target=self.animate_bar, args=(acc, color), daemon=True).start()

    def animate_bar(self, acc, color):
        self.accuracy_bar.delete("all")
        for i in range(1, int(acc) + 1):
            width = int((i / 100) * 300)
            self.accuracy_bar.delete("bar")
            self.accuracy_bar.create_rectangle(0, 0, width, 30, fill=color, width=0, tags="bar")
            self.accuracy_bar.create_text(150, 15, text=f"{i}%", fill="white",
                                          font=("Segoe UI", 12, "bold"), tags="text")
            time.sleep(0.01)


# ==========================================================
#                         RUN APP
# ==========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureChecker(root)
    root.mainloop()
