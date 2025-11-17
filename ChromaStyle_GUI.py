import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox

try:
    from process import load_seg_model, get_palette, generate_mask
    from options import opt
except ImportError:
    messagebox.showerror("Error", "Required files are missing: 'process.py', 'options.py'")
    exit()

TOP_MASK_ID = 1
BOTTOM_MASK_ID = 2

def change_color(image_bgr, mask_gray, target_color_bgr):
    target_color_bgr_np = np.uint8([[target_color_bgr]])
    target_color_hsv = cv2.cvtColor(target_color_bgr_np, cv2.COLOR_BGR2HSV)
    target_hue = target_color_hsv[0][0][0]
    target_saturation = target_color_hsv[0][0][1]

    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    h, s, v = cv2.split(hsv_image)
    
    h_new = np.where(mask_gray > 0, target_hue, h)
    s_new = np.where(mask_gray > 0, target_saturation, s)

    final_hsv = cv2.merge([h_new, s_new, v])

    final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return final_bgr

class ChromaStyleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChromaStyle - Team 17")
        self.root.geometry("1000x700")

        self.original_image_path = None
        self.original_pil_image = None
        self.original_cv_image = None
        self.final_cv_image = None
        
        self.top_color_rgb = (0, 0, 255)
        self.bottom_color_rgb = (0, 255, 0)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = 'model/cloth_segm.pth'
        try:
            self.net = load_seg_model(self.checkpoint_path, device=self.device)
            self.palette = get_palette(4)
            print(f"Model loaded successfully (Device: {self.device})")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model: {e}\nCheck 'model/cloth_segm.pth'")
            root.destroy()
            return

        control_frame = tk.Frame(root, height=100)
        control_frame.pack(fill=tk.X, pady=10)

        self.btn_load = tk.Button(control_frame, text="1. Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.btn_top_color = tk.Button(control_frame, text="2. Pick Top Color", command=self.pick_top_color)
        self.btn_top_color.pack(side=tk.LEFT, padx=10)
        self.top_color_label = tk.Label(control_frame, text="■", fg=f"#{self.top_color_rgb[0]:02x}{self.top_color_rgb[1]:02x}{self.top_color_rgb[2]:02x}")
        self.top_color_label.pack(side=tk.LEFT)

        self.btn_bottom_color = tk.Button(control_frame, text="3. Pick Bottom Color", command=self.pick_bottom_color)
        self.btn_bottom_color.pack(side=tk.LEFT, padx=10)
        self.bottom_color_label = tk.Label(control_frame, text="■", fg=f"#{self.bottom_color_rgb[0]:02x}{self.bottom_color_rgb[1]:02x}{self.bottom_color_rgb[2]:02x}")
        self.bottom_color_label.pack(side=tk.LEFT)

        self.btn_apply = tk.Button(control_frame, text="4. Apply Color Change", command=self.apply_color_change, font=('Arial', 12, 'bold'))
        self.btn_apply.pack(side=tk.LEFT, padx=20)

        self.btn_save = tk.Button(control_frame, text="5. Save Result", command=self.save_image, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=10)

        image_frame = tk.Frame(root)
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.original_label = tk.Label(image_frame, text="Original Image")
        self.original_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.result_label = tk.Label(image_frame, text="Result Image")
        self.result_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not path:
            return

        self.original_image_path = path
        
        try:
            n = np.fromfile(self.original_image_path, dtype=np.uint8)
            self.original_cv_image = cv2.imdecode(n, cv2.IMREAD_COLOR)
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to load image with OpenCV: {e}")
            self.original_image_path = None
            return
            
        if self.original_cv_image is None:
            messagebox.showerror("Image Load Error", "OpenCV (imdecode) failed to load the image.")
            self.original_image_path = None
            return

        self.original_pil_image = Image.open(self.original_image_path).convert('RGB')
        
        img_tk = self.resize_image_for_display(self.original_pil_image)
        self.original_label.config(image=img_tk)
        self.original_label.image = img_tk 
        
        self.final_cv_image = None
        self.btn_save.config(state=tk.DISABLED)
        self.result_label.config(image=None)
        self.result_label.image = None

    def pick_top_color(self):
        color = colorchooser.askcolor(title="Pick a Top Color")
        if color[0]:
            self.top_color_rgb = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            self.top_color_label.config(fg=color[1])

    def pick_bottom_color(self):
        color = colorchooser.askcolor(title="Pick a Bottom Color")
        if color[0]:
            self.bottom_color_rgb = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            self.bottom_color_label.config(fg=color[1])

    def apply_color_change(self):
        if not self.original_image_path:
            messagebox.showwarning("Error", "Please load an image first.")
            return
            
        if self.original_cv_image is None:
            messagebox.showwarning("Error", "Original image (for processing) is not loaded. Please try loading the image again.")
            return

        print("Color change process started...")
        self.btn_apply.config(text="Processing...", state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            print("1/4: Generating segmentation masks...")
            generate_mask(self.original_pil_image, net=self.net, palette=self.palette, device=self.device)
            print("2/4: Mask generation complete.")

            top_mask_path = os.path.join(opt.output, 'alpha', f'{TOP_MASK_ID}.png')
            bottom_mask_path = os.path.join(opt.output, 'alpha', f'{BOTTOM_MASK_ID}.png')

            top_mask = cv2.imread(top_mask_path, cv2.IMREAD_GRAYSCALE)
            bottom_mask = cv2.imread(bottom_mask_path, cv2.IMREAD_GRAYSCALE)

            if top_mask is None or bottom_mask is None:
                raise FileNotFoundError(f"Could not find mask files in 'output/alpha/'.\n"
                                       f"Checked for {TOP_MASK_ID}.png and {BOTTOM_MASK_ID}.png")

            print("3/4: Masks loaded successfully.")
            
            top_color_bgr = [self.top_color_rgb[2], self.top_color_rgb[1], self.top_color_rgb[0]]
            bottom_color_bgr = [self.bottom_color_rgb[2], self.bottom_color_rgb[1], self.bottom_color_rgb[0]]

            temp_image = change_color(self.original_cv_image.copy(), top_mask, top_color_bgr)
            
            self.final_cv_image = change_color(temp_image, bottom_mask, bottom_color_bgr)
            print("4/4: Color change complete.")

            final_pil_image = Image.fromarray(cv2.cvtColor(self.final_cv_image, cv2.COLOR_BGR2RGB))
            
            img_tk = self.resize_image_for_display(final_pil_image)
            self.result_label.config(image=img_tk)
            self.result_label.image = img_tk
            
            self.btn_save.config(state=tk.NORMAL)
            
            messagebox.showinfo("Success", "Color change applied!")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.final_cv_image = None
            self.btn_save.config(state=tk.DISABLED)
        finally:
            self.btn_apply.config(text="4. Apply Color Change", state=tk.NORMAL)

    def save_image(self):
        if self.final_cv_image is None:
            messagebox.showwarning("Save Error", "There is no result image to save.")
            return

        original_name = os.path.basename(self.original_image_path)
        name, ext = os.path.splitext(original_name)
        default_name = f"{name}_chroma{ext}"

        filepath = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"),
                       ("JPEG Image", "*.jpg"),
                       ("All Files", "*.*")]
        )

        if not filepath:
            return

        try:
            ext = os.path.splitext(filepath)[1]
            is_success, buffer = cv2.imencode(ext, self.final_cv_image)
            
            if not is_success:
                raise Exception("cv2.imencode failed to encode the image.")

            with open(filepath, 'wb') as f:
                f.write(buffer)
            
            messagebox.showinfo("Save Successful", f"Image saved successfully to:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image: {e}")

    def resize_image_for_display(self, pil_image):
        w, h = pil_image.size
        
        max_w = self.original_label.winfo_width() - 20
        max_h = self.original_label.winfo_height() - 20
        
        if max_w < 50: max_w = self.root.winfo_width() // 2 - 20
        if max_h < 50: max_h = self.root.winfo_height() - 120

        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        return ImageTk.PhotoImage(pil_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChromaStyleApp(root)
    
    def initial_load():
        if app.original_label.winfo_width() < 50:
            root.after(100, initial_load)
        
    root.after(100, initial_load)
    root.mainloop()