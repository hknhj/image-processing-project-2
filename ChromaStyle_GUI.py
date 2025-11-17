import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk  # PIL for Tkinter <-> OpenCV image conversion
import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox

try:
    from process import load_seg_model, get_palette, generate_mask
    from options import opt  # 'opt.output' gives us the output folder path
except ImportError:
    messagebox.showerror("Error", "Required files are missing: 'process.py', 'options.py'")
    exit()

TOP_MASK_ID = 1      # Manually set: e.g., '1.png' from output/alpha is the top
BOTTOM_MASK_ID = 2   # Manually set: e.g., '2.png' from output/alpha is the bottom

def change_color(image_bgr, mask_gray, target_color_bgr):
    """
    Changes the color of a masked area in an image while preserving texture.
    
    This function uses the HSV color space. It replaces the Hue (H) and
    Saturation (S) of the masked pixels with the target color's H and S,
    but keeps the original Value (V) to preserve shadows, highlights, and texture.

    Args:
        image_bgr (np.ndarray): The original image in OpenCV's BGR format.
        mask_gray (np.ndarray): The grayscale mask (0-255). 
                                Pixels > 0 will be recolored.
        target_color_bgr (list): The target color in BGR format, e.g., [255, 0, 0] for Blue.

    Returns:
        np.ndarray: The new image with the color-changed area in BGR format.
    """
    
    # 1. Convert the single-pixel target BGR color to HSV
    # We create a 1x1 pixel image to correctly convert the color
    target_color_bgr_np = np.uint8([[target_color_bgr]])
    target_color_hsv = cv2.cvtColor(target_color_bgr_np, cv2.COLOR_BGR2HSV)
    target_hue = target_color_hsv[0][0][0]
    target_saturation = target_color_hsv[0][0][1]

    # 2. Convert the original BGR image to HSV
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # 3. Split the HSV image into H, S, V channels
    h, s, v = cv2.split(hsv_image)
    
    # 4. Use the mask to create new H and S channels
    # Where the mask is white (mask_gray > 0), use the target H and S.
    # Otherwise, use the original H and S.
    h_new = np.where(mask_gray > 0, target_hue, h)
    s_new = np.where(mask_gray > 0, target_saturation, s)
    # The original V (Value) channel is kept to preserve texture and lighting.

    # 5. Merge the new H, S, and original V channels back together
    final_hsv = cv2.merge([h_new, s_new, v])

    # 6. Convert the final HSV image back to BGR
    final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return final_bgr

class ChromaStyleApp:
    def __init__(self, root):
        """
        Initializes the main GUI application window.
        """
        self.root = root
        self.root.title("ChromaStyle - Team 17")
        self.root.geometry("1000x700") # Set window size

        # --- State Variables ---
        # These variables store the application's state
        self.original_image_path = None
        self.original_pil_image = None # PIL Image (for Tkinter display)
        self.original_cv_image = None  # OpenCV Image (for processing)
        
        # Default colors (RGB format for Tkinter)
        self.top_color_rgb = (0, 0, 255)     # Default: Blue
        self.bottom_color_rgb = (0, 255, 0) # Default: Green

        # --- Load Deep Learning Model ---
        # Load the segmentation model from process.py on startup
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

        # --- GUI Layout ---
        
        # 1. Top Control Frame (for buttons)
        control_frame = tk.Frame(root, height=100)
        control_frame.pack(fill=tk.X, pady=10)

        # Button: Load Image
        self.btn_load = tk.Button(control_frame, text="1. Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=10)

        # Button: Pick Top Color
        self.btn_top_color = tk.Button(control_frame, text="2. Pick Top Color", command=self.pick_top_color)
        self.btn_top_color.pack(side=tk.LEFT, padx=10)
        self.top_color_label = tk.Label(control_frame, text="■", fg=f"#{self.top_color_rgb[0]:02x}{self.top_color_rgb[1]:02x}{self.top_color_rgb[2]:02x}")
        self.top_color_label.pack(side=tk.LEFT) # Shows the selected color

        # Button: Pick Bottom Color
        self.btn_bottom_color = tk.Button(control_frame, text="3. Pick Bottom Color", command=self.pick_bottom_color)
        self.btn_bottom_color.pack(side=tk.LEFT, padx=10)
        self.bottom_color_label = tk.Label(control_frame, text="■", fg=f"#{self.bottom_color_rgb[0]:02x}{self.bottom_color_rgb[1]:02x}{self.bottom_color_rgb[2]:02x}")
        self.bottom_color_label.pack(side=tk.LEFT) # Shows the selected color

        # Button: Apply Changes
        self.btn_apply = tk.Button(control_frame, text="4. Apply Color Change", command=self.apply_color_change, font=('Arial', 12, 'bold'))
        self.btn_apply.pack(side=tk.LEFT, padx=20)

        # 2. Bottom Image Frame (for Original vs. Result)
        image_frame = tk.Frame(root)
        image_frame.pack(fill=tk.BOTH, expand=True)

        # Label to display the Original Image
        self.original_label = tk.Label(image_frame, text="Original Image")
        self.original_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Label to display the Result Image
        self.result_label = tk.Label(image_frame, text="Result Image")
        self.result_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

    def load_image(self):
        """
        Opens a file dialog for the user to select an image.
        Loads the image into both PIL (for display) and OpenCV (for processing) formats.
        """
        # Open file dialog to choose an image
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not path:
            return # User cancelled

        self.original_image_path = path
        
        # Load image for OpenCV (processing) - BGR format
        self.original_cv_image = cv2.imread(self.original_image_path)
        # Load image for PIL (display) - RGB format
        self.original_pil_image = Image.open(self.original_image_path).convert('RGB')
        
        # Resize and display the original image
        img_tk = self.resize_image_for_display(self.original_pil_image)
        self.original_label.config(image=img_tk)
        self.original_label.image = img_tk # Keep a reference to prevent garbage collection

    def pick_top_color(self):
        """
        Opens a color chooser dialog for the 'Top' color.
        Updates the state and the color preview label.
        """
        color = colorchooser.askcolor(title="Pick a Top Color")
        if color[0]: # color[0] is the (R, G, B) tuple
            self.top_color_rgb = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            self.top_color_label.config(fg=color[1]) # color[1] is the hex code

    def pick_bottom_color(self):
        """
        Opens a color chooser dialog for the 'Bottom' color.
        Updates the state and the color preview label.
        """
        color = colorchooser.askcolor(title="Pick a Bottom Color")
        if color[0]:
            self.bottom_color_rgb = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            self.bottom_color_label.config(fg=color[1])

    def apply_color_change(self):
        # 1. Validation
        if not self.original_image_path:
            messagebox.showwarning("Error", "Please load an image first.")
            return

        print("Color change process started...")
        # Disable button during processing
        self.btn_apply.config(text="Processing...", state=tk.DISABLED)
        self.root.update_idletasks() # Force GUI update

        try:
            print("1/4: Generating segmentation masks...")
            # This function takes a PIL image and saves masks to 'output/alpha/'
            generate_mask(self.original_pil_image, net=self.net, palette=self.palette, device=self.device)
            print("2/4: Mask generation complete.")

            # Build paths to the mask files using the IDs we defined at the top
            top_mask_path = os.path.join(opt.output, 'alpha', f'{TOP_MASK_ID}.png')
            bottom_mask_path = os.path.join(opt.output, 'alpha', f'{BOTTOM_MASK_ID}.png')

            # Load masks using OpenCV (grayscale)
            top_mask = cv2.imread(top_mask_path, cv2.IMREAD_GRAYSCALE)
            bottom_mask = cv2.imread(bottom_mask_path, cv2.IMREAD_GRAYSCALE)

            # Check if masks were loaded successfully
            if top_mask is None or bottom_mask is None:
                raise FileNotFoundError(f"Could not find mask files in 'output/alpha/'.\n"
                                       f"Checked for {TOP_MASK_ID}.png and {BOTTOM_MASK_ID}.png")

            print("3/4: Masks loaded successfully.")
            
            # Convert Tkinter's RGB color to OpenCV's BGR color
            top_color_bgr = [self.top_color_rgb[2], self.top_color_rgb[1], self.top_color_rgb[0]]
            bottom_color_bgr = [self.bottom_color_rgb[2], self.bottom_color_rgb[1], self.bottom_color_rgb[0]]

            # Apply 'top' color change
            temp_image = change_color(self.original_cv_image.copy(), top_mask, top_color_bgr)
            # Apply 'bottom' color change *on top of the previous result*
            final_cv_image = change_color(temp_image, bottom_mask, bottom_color_bgr)
            print("4/4: Color change complete.")

            # Convert final OpenCV BGR image back to PIL RGB format for display
            final_pil_image = Image.fromarray(cv2.cvtColor(final_cv_image, cv2.COLOR_BGR2RGB))
            
            # Resize and show in the GUI
            img_tk = self.resize_image_for_display(final_pil_image)
            self.result_label.config(image=img_tk)
            self.result_label.image = img_tk # Keep reference
            
            messagebox.showinfo("Success", "Color change applied!")

        except Exception as e:
            # Show any errors to the user
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
        finally:
            # Re-enable the button whether it succeeded or failed
            self.btn_apply.config(text="4. Apply Color Change", state=tk.NORMAL)

    def resize_image_for_display(self, pil_image):
        """
        Resizes a PIL image to fit within the GUI display area,
        maintaining its aspect ratio.
        """
        w, h = pil_image.size
        # Calculate max width/height based on window size
        max_w = self.root.winfo_width() // 2 - 20 # Half the window width
        max_h = self.root.winfo_height() - 120 # Window height minus controls

        # If image is larger than the display area
        if w > max_w or h > max_h:
            # Calculate the new size ratio
            ratio = min(max_w / w, max_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        # Convert PIL image to a Tkinter-compatible PhotoImage
        return ImageTk.PhotoImage(pil_image)

if __name__ == "__main__":
    """
    This is the main entry point of the script.
    It creates the Tkinter window and starts the application.
    """
    root = tk.Tk()
    app = ChromaStyleApp(root)
    root.mainloop() # Start the GUI event loop