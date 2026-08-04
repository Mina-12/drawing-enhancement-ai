import pickle
import os.path

import tkinter.messagebox
from tkinter import *
from tkinter import Tk, Canvas, Button, simpledialog, filedialog, NW
import tkinter as tk
from tkinter import ttk
from tkinter import BOTH, BOTTOM, X, YES


import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageTk







from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class DrawingClassifier:

    def __init__(self):
        self.class1, self.class2 = None, None
        self.class1_counter, self.class2_counter = None, None
        self.clf = None
        self.proj_name = None
        self.root = None
        self.image1 = None
        self.uploaded_photo = None  
        self.enhanced_image_tk = None



        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 0.1

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.proj_name = "dogs_vs_cats"
        if os.path.exists(self.proj_name):
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class1_counter = data['c1c']
            self.class2_counter = data['c2c']
            self.clf = data['clf']
            self.proj_name = data['pname']
        else:
            self.class1 = simpledialog.askstring("Class 1", "What is the first class called?", parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "What is the second class called?", parent=msg)

            self.class1_counter = 1
            self.class2_counter = 1

            self.clf = LinearSVC()

            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.chdir("..")

    def init_gui(self):
     WIDTH, HEIGHT = 500, 500
     WHITE = (255, 255, 255)

     self.root = tk.Tk()
     self.root.title(f"Drawing Enhancer - {self.proj_name}")

     # Canvas Area
     self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="white", relief=tk.RIDGE, bd=2)
     self.canvas.pack(side=tk.TOP, expand=YES, fill=BOTH, padx=10, pady=10)
     self.canvas.bind("<B1-Motion>", self.paint)

     self.image1 = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
     self.draw = ImageDraw.Draw(self.image1)

    # Button Frame
     btn_frame = ttk.Frame(self.root, padding=10)
     btn_frame.pack(side=BOTTOM, fill=X)

    # Drawing Controls
     draw_label = ttk.Label(btn_frame, text="Drawing", font=("Arial", 10, "bold"))
     draw_label.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))

     ttk.Button(btn_frame, text="Brush -", command=self.brushminus).grid(row=1, column=0, sticky="ew", padx=2)
     ttk.Button(btn_frame, text="Clear", command=self.clear).grid(row=1, column=1, sticky="ew", padx=2)
     ttk.Button(btn_frame, text="Brush +", command=self.brushplus).grid(row=1, column=2, sticky="ew", padx=2)

     # Class Buttons
     ttk.Button(btn_frame, text=self.class1, command=lambda: self.save(1)).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
     ttk.Button(btn_frame, text=self.class2, command=lambda: self.save(2)).grid(row=2, column=1, sticky="ew", padx=2, pady=2)
     ttk.Button(btn_frame, text="Upload Sketch", command=self.upload_sketch).grid(row=2, column=2, sticky="ew", padx=2, pady=2)

     # Model Section
     model_label = ttk.Label(btn_frame, text="Model Controls", font=("Arial", 10, "bold"))
     model_label.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 5))

     ttk.Button(btn_frame, text="Train Model", command=self.train_model).grid(row=4, column=0, sticky="ew", padx=2)
     ttk.Button(btn_frame, text="Save Model", command=self.save_model).grid(row=4, column=1, sticky="ew", padx=2)
     ttk.Button(btn_frame, text="Load Model", command=self.load_model).grid(row=4, column=2, sticky="ew", padx=2)

     ttk.Button(btn_frame, text="Change Model", command=self.rotate_model).grid(row=5, column=0, sticky="ew", padx=2, pady=2)
     ttk.Button(btn_frame, text="Save Everything", command=self.save_everything).grid(row=5, column=1, sticky="ew", padx=2, pady=2)

     # Prediction & Enhancement
     action_label = ttk.Label(btn_frame, text="Inference", font=("Arial", 10, "bold"))
     action_label.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 5))

     ttk.Button(btn_frame, text="Predict", command=self.predict).grid(row=7, column=0, sticky="ew", padx=2)
     ttk.Button(btn_frame, text="Enhance", command=self.enhance).grid(row=7, column=1, sticky="ew", padx=2)

     # Status Bar
     self.status_label = ttk.Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}", anchor="center")
     self.status_label.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 0))

     for i in range(3):
         btn_frame.columnconfigure(i, weight=1)

     self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
     self.root.attributes("-topmost", True)
     self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    def save(self, class_num):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((500, 500), PIL.Image.LANCZOS)

        if class_num == 1:
            img.save(f"{self.proj_name}/{self.class1}/{self.class1_counter}.png", "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.proj_name}/{self.class2}/{self.class2_counter}.png", "PNG")
            self.class2_counter += 1

        self.clear()

    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def brushplus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def train_model(self):
        img_list = np.array([])  # List to hold image data
        class_list = np.array([])  # List to hold class labels

        # Load images from the 'dogs' folder and assign class 1 (dogs)
        for filename in os.listdir(f"{self.proj_name}/{self.class1}"):
            if filename.endswith(".jpg"):  # Ensure it's a .png image
                img = cv.imread(f"{self.proj_name}/{self.class1}/{filename}")[:, :, 0]
                img = cv.resize(img, (50, 50))  # Resize to 50x50 to match the drawing input size
                img = img.reshape(2500)  # Flatten the image to a 1D array
                img_list = np.append(img_list, [img])  # Add image data to the list
                class_list = np.append(class_list, 1)  # Class 1 for dogs

        # Load images from the 'cats' folder and assign class 2 (cats)
        for filename in os.listdir(f"{self.proj_name}/{self.class2}"):
            if filename.endswith(".jpg"):  # Ensure it's a .png image
                img = cv.imread(f"{self.proj_name}/{self.class2}/{filename}")[:, :, 0]
                img = cv.resize(img, (50, 50))  # Resize to 50x50 to match the drawing input size
                img = img.reshape(2500)  # Flatten the image to a 1D array
                img_list = np.append(img_list, [img])  # Add image data to the list
                class_list = np.append(class_list, 2)  # Class 2 for cats

        img_list = img_list.reshape(len(class_list), 2500)  # Reshape list to match the number of images

        self.clf.fit(img_list, class_list)  # Train the classifier
        tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully trained!", parent=self.root)



    def predict(self):
     if self.image1 is None:
         tkinter.messagebox.showwarning("No Image", "Please upload a sketch before predicting.")
         return

     # Convert PIL image (self.image1) to OpenCV format (NumPy array)
     img = np.array(self.image1.convert("L"))  # Grayscale

     # Resize to match the classifier input size
     img = cv.resize(img, (50, 50))
     img = img.reshape(1, -1)

     # Predict
     prediction = self.clf.predict(img)

     # Show result
     if prediction[0] == 1:
         tkinter.messagebox.showinfo("Prediction", f"The sketch is probably  {self.class1}")
     elif prediction[0] == 2:
         tkinter.messagebox.showinfo("Prediction", f"The sketch is probably  {self.class2}")

     
     temp_path = os.path.join("sketch.png")
     self.image1.save(temp_path)
     self.temp_sketch_path = temp_path 


          

    def upload_sketch(self):
     file_path = filedialog.askopenfilename(
         filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")],
         parent=self.root  # Ensure dialog is tied to main window
     )
     if not file_path:
          return

     # Open and resize image
     img = Image.open(file_path).convert("RGB")
     img = img.resize((500, 500), Image.LANCZOS)

     # Store image and context for prediction
     self.image1 = img
     self.draw = ImageDraw.Draw(self.image1)

     # Create PhotoImage using the current root
     self.uploaded_photo = ImageTk.PhotoImage(self.image1, master=self.root)

     # Display image on canvas
     self.canvas.delete("all")  # clear canvas first
     self.canvas.create_image(0, 0, anchor=NW, image=self.uploaded_photo)

     # Prevent garbage collection
     self.canvas.image = self.uploaded_photo






    def enhance(self):
     try:
         # Check if prediction and image path are available
         if not hasattr(self, "temp_sketch_path") or not os.path.exists(self.temp_sketch_path):
             tkinter.messagebox.showwarning("Enhance Error", "Please predict the sketch before enhancing.")
             return

         if not hasattr(self, "class1") or not hasattr(self, "class2"):
             tkinter.messagebox.showerror("Error", "Prediction classes are not set.")
             return

         # Determine class name based on last prediction
         img = np.array(self.image1.convert("L"))
         img = cv.resize(img, (50, 50)).reshape(1, -1)
         prediction = self.clf.predict(img)

         if prediction[0] == 1:
             predicted_class = self.class1
         elif prediction[0] == 2:
             predicted_class = self.class2
         else:
             tkinter.messagebox.showerror("Prediction Error", "Could not determine predicted class.")
             return

         # Show loading message
         loading = tkinter.Label(self.root, text="Enhancing sketch, please wait...", font=("Arial", 12))
         loading.pack()
         self.root.update()
 
         # === Load the corresponding generator model ===
         from models.generator import UNetGenerator
         import torch
         from torchvision import transforms
         

         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         checkpoint_map = {
             self.class1.lower(): "checkpoints_cats/gen_epoch400.pth",  # cat model
             self.class2.lower(): "checkpoints_dogs/gen_epoch500.pth",  # dog model
         }

         checkpoint_path = checkpoint_map.get(predicted_class.lower())
         if not checkpoint_path or not os.path.exists(checkpoint_path):
             loading.destroy()
             tkinter.messagebox.showerror("Model Error", f"No enhancement model found for: {predicted_class}")
             return

         # Load model
         gen = UNetGenerator(input_channels=1, output_channels=3).to(device)
         gen.load_state_dict(torch.load(checkpoint_path, map_location=device))
         gen.eval()

        # Transform the input
         transform = transforms.Compose([
             transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])
         ])

         sketch = Image.open(self.temp_sketch_path).convert("L")
         sketch_tensor = transform(sketch).unsqueeze(0).to(device)

         with torch.no_grad():
             enhanced_tensor = gen(sketch_tensor).squeeze(0).cpu()

         enhanced_tensor = (enhanced_tensor * 0.5 + 0.5).clamp(0, 1)
         enhanced_pil = transforms.ToPILImage()(enhanced_tensor)

         # Resize both images to the same size for consistency
         original = self.image1.convert("RGB").resize((256, 256), Image.LANCZOS)
         enhanced = enhanced_pil.resize((256, 256), Image.LANCZOS)

         # Create a new image wide enough to hold both side by side
         combined_width = original.width + enhanced.width
         combined = Image.new("RGB", (combined_width, original.height))

         # Paste original and enhanced side by side
         combined.paste(original, (0, 0))
         combined.paste(enhanced, (original.width, 0))

        # Save side-by-side image 
         combined_path = os.path.join("before_vs_after.png")
         combined.save(combined_path)

        # Also optionally save enhanced image alone
         enhanced_path = os.path.join("enhanced.png")
         enhanced.save(enhanced_path)


         # Show on canvas 
         enhanced_pil = enhanced_pil.resize((500, 500), Image.LANCZOS)
         self.enhanced_image_tk = ImageTk.PhotoImage(enhanced_pil, master=self.root)
         self.canvas.delete("all")
         self.canvas.create_image(0, 0, anchor=NW, image=self.enhanced_image_tk)

         loading.destroy()
         tkinter.messagebox.showinfo("Enhancement Complete", f"Enhanced sketch as a {predicted_class}!")

     except Exception as e:
         tkinter.messagebox.showerror("Enhancement Error", f"Something went wrong:\n{e}")








    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()

        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.clf, f)
        tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully saved!", parent=self.root)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully loaded!", parent=self.root)

    def save_everything(self):
        data = {"c1": self.class1, "c2": self.class2, "c1c": self.class1_counter,
                "c2c": self.class2_counter, "clf": self.clf, "pname": self.proj_name}
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("Drawing Classifier", "Project successfully saved!", parent=self.root)

    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?", "Do you want to save your work?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()


DrawingClassifier()